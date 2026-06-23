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

#include "pypto/ir/transforms/op_conversion_registry.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

using tile_conversion_utils::MakeShapeTuple;
using tile_conversion_utils::MakeZeroOffsets;

namespace {

bool IsConstOne(const ExprPtr& expr) { return IsConstValue(expr, 1); }

// Detect row-broadcast pattern: [M, N] op [M, 1] or [M, 1] op [M, N]
// Returns {wider_arg_idx, narrower_arg_idx} if broadcast detected, empty otherwise
std::pair<int, int> DetectRowBroadcast(const std::vector<ExprPtr>& args) {
  auto type0 = As<TileType>(args[0]->GetType());
  auto type1 = As<TileType>(args[1]->GetType());
  if (!type0 || !type1) return {-1, -1};
  if (type0->shape_.size() != 2 || type1->shape_.size() != 2) return {-1, -1};

  bool rhs_is_col_vec = IsConstOne(type1->shape_[1]) && !IsConstOne(type0->shape_[1]);
  bool lhs_is_col_vec = IsConstOne(type0->shape_[1]) && !IsConstOne(type1->shape_[1]);

  if (rhs_is_col_vec) return {0, 1};
  if (lhs_is_col_vec) return {1, 0};
  return {-1, -1};
}

template <typename T>
T GetKwargOr(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
             const T& default_value) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  return default_value;
}

}  // namespace

OpConversionRegistry& OpConversionRegistry::GetInstance() {
  static OpConversionRegistry instance;
  return instance;
}

OpConversionRegistry::OpConversionRegistry() {
  RegisterScalarAndUnaryOps();
  RegisterBroadcastAndTransformOps();
  RegisterElementwiseBinaryOps();
  RegisterMemoryOps();
  RegisterMatmulOps();
  RegisterReductionOps();
  RegisterSortOps();
  RegisterGatherOps();
  RegisterPagedGatherOps();
  RegisterScatterOps();
  RegisterCmpOps();
  RegisterDistributedOps();
}

// ============================================================================
// Scalar and unary ops: simple 1:1 tensor → tile name mapping
// ============================================================================

void OpConversionRegistry::RegisterScalarAndUnaryOps() {
  RegisterSimple("tensor.adds", "tile.adds");
  RegisterSimple("tensor.subs", "tile.subs");
  RegisterSimple("tensor.muls", "tile.muls");
  RegisterSimple("tensor.divs", "tile.divs");

  RegisterSimple("tensor.neg", "tile.neg");
  RegisterSimple("tensor.abs", "tile.abs");
  RegisterSimple("tensor.recip", "tile.recip");
  RegisterSimple("tensor.exp", "tile.exp");
  RegisterSimple("tensor.log", "tile.log");
  RegisterSimple("tensor.sin", "tile.sin");
  RegisterSimple("tensor.cos", "tile.cos");
  RegisterSimple("tensor.sqrt", "tile.sqrt");
  RegisterSimple("tensor.cast", "tile.cast");

  // tensor.rsqrt → tile.rsqrt (basic) or tile.rsqrt(src, tmp) (high-precision).
  // The tmp scratch tile is allocated via tile.create when high_precision=True.
  RegisterCustom(
      "tensor.rsqrt",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 1, span)
            << "tensor.rsqrt conversion expects 1 arg, got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        bool high_precision = GetKwargOr<bool>(kwargs, "high_precision", false);
        if (!high_precision) {
          return ConversionResult{op_reg.Create("tile.rsqrt", {input}, span)};
        }

        auto tile_type = As<TileType>(input->GetType());
        INTERNAL_CHECK_SPAN(tile_type, span)
            << "tensor.rsqrt conversion: input must be TileType after memory promotion, got "
            << input->GetType()->TypeName();

        auto shape_tuple = std::make_shared<MakeTuple>(tile_type->shape_, span);
        std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", tile_type->dtype_},
                                                                       {"target_memory", MemorySpace::Vec}};
        auto create_call = op_reg.Create("tile.create", {shape_tuple}, create_kwargs, span);

        auto tmp_var = std::make_shared<Var>("rsqrt_tmp", create_call->GetType(), span);
        std::vector<StmtPtr> prologue;
        prologue.push_back(std::make_shared<AssignStmt>(tmp_var, create_call, span));

        auto rsqrt_call = op_reg.Create("tile.rsqrt", {input, tmp_var}, span);
        return ConversionResult{std::move(prologue), rsqrt_call};
      });
}

// ============================================================================
// Broadcast and transform ops: simple 1:1 name mapping
// ============================================================================

void OpConversionRegistry::RegisterBroadcastAndTransformOps() {
  RegisterSimple("tensor.row_expand_mul", "tile.row_expand_mul");
  RegisterSimple("tensor.row_expand_div", "tile.row_expand_div");
  RegisterSimple("tensor.col_expand_mul", "tile.col_expand_mul");
  RegisterSimple("tensor.col_expand_add", "tile.col_expand_add");
  RegisterSimple("tensor.row_expand", "tile.row_expand");
  RegisterSimple("tensor.row_expand_add", "tile.row_expand_add");
  RegisterSimple("tensor.row_expand_sub", "tile.row_expand_sub");
  RegisterSimple("tensor.col_expand", "tile.col_expand");
  RegisterSimple("tensor.col_expand_sub", "tile.col_expand_sub");
  RegisterSimple("tensor.col_expand_div", "tile.col_expand_div");
  RegisterSimple("tensor.row_expand_max", "tile.row_expand_max");
  RegisterSimple("tensor.row_expand_min", "tile.row_expand_min");
  RegisterSimple("tensor.row_expand_expdif", "tile.row_expand_expdif");
  RegisterSimple("tensor.col_expand_max", "tile.col_expand_max");
  RegisterSimple("tensor.col_expand_min", "tile.col_expand_min");
  RegisterSimple("tensor.col_expand_expdif", "tile.col_expand_expdif");
  RegisterSimple("tensor.expands", "tile.expands");

  RegisterSimple("tensor.reshape", "tile.reshape");

  // tensor.transpose → tile.transpose(input, axis1, axis2). The pto.ttrans scratch is a pure
  // codegen detail, not a semantic operand: FlattenTileNdTo2D is the sole owner of scratch
  // materialization (it emits the codegen-ready 4-arg form for both 2D and per-page >2D
  // transposes, before the memory allocator runs). So the conversion emits no tmp here.
  RegisterCustom(
      "tensor.transpose",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        // The optional 4th tensor-level arg is valid_shape; it has no tile-level equivalent.
        INTERNAL_CHECK_SPAN(args.size() == 3 || args.size() == 4, span)
            << "tensor.transpose conversion expects 3 or 4 args (input, axis1, axis2[, valid_shape]), got "
            << args.size();
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        INTERNAL_CHECK_SPAN(As<TileType>(input->GetType()), span)
            << "tensor.transpose conversion: input must be TileType after memory promotion, got "
            << input->GetType()->TypeName();

        auto transpose_call = op_reg.Create("tile.transpose", {input, args[1], args[2]}, span);
        return ConversionResult{{}, transpose_call};
      });

  RegisterSimple("tensor.concat", "tile.concat");
  RegisterSimple("tensor.set_validshape", "tile.set_validshape");

  RegisterSimple("tensor.full", "tile.full");
  RegisterSimple("tensor.ci", "tile.ci");
}

// ============================================================================
// Broadcast-aware elementwise binary ops
//
// When both operands have the same shape → tile.{op}
// When one operand is [M,1] (column vector) → tile.row_expand_{op}
// ============================================================================

void OpConversionRegistry::RegisterElementwiseBinaryOps() {
  auto MakeBroadcastBinaryConv = [](const std::string& tile_op,
                                    const std::string& row_expand_op) -> ConversionFunc {
    return [tile_op, row_expand_op](const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const Span& span) -> ConversionResult {
      auto& op_reg = OpRegistry::GetInstance();
      auto [wider, narrower] = DetectRowBroadcast(args);
      if (wider >= 0) {
        return ConversionResult{op_reg.Create(row_expand_op, {args[wider], args[narrower]}, span)};
      }
      if (kwargs.empty()) {
        return ConversionResult{op_reg.Create(tile_op, args, span)};
      }
      return ConversionResult{op_reg.Create(tile_op, args, kwargs, span)};
    };
  };

  RegisterCustom("tensor.add", MakeBroadcastBinaryConv("tile.add", "tile.row_expand_add"));
  RegisterCustom("tensor.sub", MakeBroadcastBinaryConv("tile.sub", "tile.row_expand_sub"));
  RegisterCustom("tensor.mul", MakeBroadcastBinaryConv("tile.mul", "tile.row_expand_mul"));
  RegisterCustom("tensor.div", MakeBroadcastBinaryConv("tile.div", "tile.row_expand_div"));
  // tensor.maximum/minimum dispatch by rhs type:
  //   tensor rhs → tile.maximum/minimum
  //   scalar rhs → tile.maximums/minimums
  // There is no tensor.maximums/minimums front-end op — the unified tensor op
  // is rewritten here based on the rhs operand type.
  auto MakeMinMaxConv = [](const std::string& tile_op, const std::string& tile_scalar_op) -> ConversionFunc {
    return [tile_op, tile_scalar_op](const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const Span& span) -> ConversionResult {
      INTERNAL_CHECK_SPAN(args.size() == 2, span)
          << "tensor.maximum/minimum conversion expects 2 args, got " << args.size();
      auto& op_reg = OpRegistry::GetInstance();
      const std::string& chosen = As<ScalarType>(args[1]->GetType()) ? tile_scalar_op : tile_op;
      if (kwargs.empty()) {
        return ConversionResult{op_reg.Create(chosen, args, span)};
      }
      return ConversionResult{op_reg.Create(chosen, args, kwargs, span)};
    };
  };
  RegisterCustom("tensor.maximum", MakeMinMaxConv("tile.maximum", "tile.maximums"));
  RegisterCustom("tensor.minimum", MakeMinMaxConv("tile.minimum", "tile.minimums"));
}

// ============================================================================
// Memory ops: slice, assemble, create, fillpad, scatter_update, read, write
// ============================================================================

void OpConversionRegistry::RegisterMemoryOps() {
  // tensor.slice → tile.load (gm_tensor) or tile.slice (local_tensor)
  RegisterCustom(
      "tensor.slice",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 3 || args.size() == 4, span)
            << "tensor.slice conversion expects 3 or 4 args (tensor, shape, offset[, valid_shape])";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];
        const auto& shape = args[1];
        const auto& offset = args[2];

        // Extract pad_value kwarg (if any) to forward to the emitted tile.slice.
        std::vector<std::pair<std::string, std::any>> forward_kwargs;
        for (const auto& kv : kwargs) {
          if (kv.first == "pad_value") {
            forward_kwargs.push_back(kv);
            break;
          }
        }

        // ``AsTensorTypeLike`` matches both ``TensorType`` and
        // ``DistributedTensorType`` (issue #1694): inside an InCore scope a
        // window is just this rank's local GM, so slicing it lowers to a
        // ``tile.load`` exactly like a plain tensor slice. The exact-kind
        // ``As<TensorType>`` would miss the window and trip the unreachable
        // guard below.
        auto tensor_type = AsTensorTypeLike(input->GetType());
        auto tile_type = As<TileType>(input->GetType());

        if (tensor_type) {
          // The tile.load path does not currently accept pad_value. If the user set
          // pad_value on a tensor.slice over a TensorType input, the pad intent is
          // lost here — a follow-up tile.fillpad is the workaround until tile.load
          // grows its own pad_value kwarg.
          auto valid_shapes = (args.size() == 4) ? args[3] : shape;
          std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                       {"transpose", false}};
          auto load_call =
              op_reg.Create("tile.load", {input, offset, shape, valid_shapes}, load_kwargs, span);
          return ConversionResult{load_call};
        }

        if (tile_type) {
          std::vector<ExprPtr> slice_args = {input, shape, offset};
          if (args.size() == 4) {
            slice_args.push_back(args[3]);
          }
          auto slice_call = op_reg.Create("tile.slice", slice_args, forward_kwargs, span);
          return ConversionResult{slice_call};
        }

        INTERNAL_UNREACHABLE_SPAN(span)
            << "tensor.slice conversion: unexpected input type: " << input->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // tensor.assemble → tile.store or tile.assemble depending on types
  RegisterCustom(
      "tensor.assemble",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 3, span)
            << "tensor.assemble conversion expects 3 args (target, source, offset)";
        auto& op_reg = OpRegistry::GetInstance();

        const auto& target = args[0];
        const auto& source = args[1];
        const auto& offset = args[2];

        // ``AsTensorTypeLike`` matches a ``DistributedTensorType`` window the
        // same way it does a plain tensor (issue #1694): a slice-assign whose
        // target is a local window must lower to ``tile.store`` into that
        // window, not fall through to the unconverted fallback (which the
        // InCore verifier then rejects).
        auto source_tile_type = As<TileType>(source->GetType());
        auto target_tensor_type = AsTensorTypeLike(target->GetType());
        auto target_tile_type = As<TileType>(target->GetType());

        // Optional atomic-add combine mode. Valid only on the GM-store path
        // (tile source + tensor target) — a tile-to-tile assemble has no
        // global-memory destination to atomically accumulate into.
        int atomic = GetKwargOr<int>(kwargs, "atomic", static_cast<int>(AtomicType::kNone));
        const bool atomic_add = atomic == static_cast<int>(AtomicType::kAdd);
        constexpr const char* kAtomicTileToTileMsg =
            "tensor.assemble with atomic=AtomicType.Add requires a global-memory destination "
            "(a function output tensor), but this assemble targets an on-chip tile";

        if (source_tile_type && target_tensor_type) {
          if (atomic_add) {
            std::vector<std::pair<std::string, std::any>> store_kw = {{"atomic", atomic}};
            return ConversionResult{op_reg.Create("tile.store", {source, offset, target}, store_kw, span)};
          }
          return ConversionResult{op_reg.Create("tile.store", {source, offset, target}, span)};
        }

        if (source_tile_type && target_tile_type) {
          INTERNAL_CHECK_SPAN(!atomic_add, span) << kAtomicTileToTileMsg;
          auto assemble_call = op_reg.Create("tile.assemble", {target, source, offset}, span);
          return ConversionResult{assemble_call};
        }

        if (target_tile_type && !source_tile_type) {
          INTERNAL_CHECK_SPAN(!atomic_add, span) << kAtomicTileToTileMsg;
          // A window (DistributedTensorType) source stages into the tile target
          // through the same local tile.load path as a plain tensor (issue #1694).
          auto source_tensor_type = AsTensorTypeLike(source->GetType());
          INTERNAL_CHECK_SPAN(source_tensor_type, span)
              << "tensor.assemble: source must be TensorType, DistributedTensorType or TileType, but got "
              << source->GetType()->TypeName();
          std::vector<StmtPtr> prologue;
          auto offsets_load = MakeZeroOffsets(source_tensor_type->shape_.size(), span);
          auto shapes = MakeShapeTuple(source_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", MemorySpace::Vec},
                                                                   {"transpose", false}};
          auto load_call = op_reg.Create("tile.load", {source, offsets_load, shapes, shapes}, load_kw, span);
          auto source_tile_var = std::make_shared<Var>("assemble_src", load_call->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(source_tile_var, load_call, span));

          auto assemble_call = op_reg.Create("tile.assemble", {target, source_tile_var, offset}, span);
          return ConversionResult{std::move(prologue), assemble_call};
        }

        if (kwargs.empty()) {
          return ConversionResult{op_reg.Create("tensor.assemble", args, span)};
        }
        return ConversionResult{op_reg.Create("tensor.assemble", args, kwargs, span)};
      });

  // tensor.scatter_update → tile.scatter (whole-row scatter via flat indices).
  //   input [m, d], index [b, s] (row numbers), src [b*s, d]: input[index.flat[k]] = src[k].
  //   pto.tscatter writes per-element: dst.flat[flat_idx[k, c]] = src[k, c], so each src
  //   row k is written whole into dst row index.flat[k] when flat_idx[k, c] = index.flat[k]*d + c.
  //   Build flat_idx = ci([n,d]) + (index.flat - k)*d, where ci[k,c]=k*d+c, then reconstruct the
  //   DPS row-preserve with the same zeroed-scatter + mask + select blend as tensor.scatter.
  std::unordered_map<size_t, InputSpaceReq> scatter_update_input_reqs = {
      {0, {MemorySpace::Vec, std::nullopt}},
      {1, {MemorySpace::Vec, std::nullopt}},
      {2, {MemorySpace::Vec, std::nullopt}},
  };
  auto scatter_update_conv = [](const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const Span& span) -> ConversionResult {
    INTERNAL_CHECK_SPAN(args.size() == 3, span)
        << "tensor.scatter_update conversion expects 3 args (input, index, src)";
    auto& op_reg = OpRegistry::GetInstance();
    auto input_tile = As<TileType>(args[0]->GetType());
    auto idx_tile = As<TileType>(args[1]->GetType());
    auto src_tile = As<TileType>(args[2]->GetType());
    INTERNAL_CHECK_SPAN(input_tile && idx_tile && src_tile, span)
        << "tensor.scatter_update conversion: input/index/src must be Vec tiles after bridge";
    // 4D input/src is accepted by the op's type deduction but not yet lowered, so a 4D call
    // type-checks and would otherwise hit an internal error here — surface it as a user error.
    CHECK_SPAN(input_tile->shape_.size() == 2 && src_tile->shape_.size() == 2, span)
        << "scatter_update: only 2D input/src is currently supported in lowering, got input rank "
        << input_tile->shape_.size() << " and src rank " << src_tile->shape_.size();
    auto src_rows = As<ConstInt>(src_tile->shape_[0]);
    auto idx_b = As<ConstInt>(idx_tile->shape_[0]);
    auto idx_s = As<ConstInt>(idx_tile->shape_[1]);
    auto dst_rows = As<ConstInt>(input_tile->shape_[0]);
    auto dst_cols = As<ConstInt>(input_tile->shape_[1]);
    INTERNAL_CHECK_SPAN(src_rows && idx_b && idx_s && dst_rows && dst_cols, span)
        << "tensor.scatter_update conversion requires static shapes for index expansion";
    const int64_t n = src_rows->value_;  // b*s flattened scatter rows
    // src rows must equal the flattened index size, since each src row k is written whole
    // into dst[index.flat[k]] and flat_idx is built over the [n, d] template (n = b * s).
    INTERNAL_CHECK_SPAN(n == idx_b->value_ * idx_s->value_, span)
        << "tensor.scatter_update conversion: src rows (" << n
        << ") must match index size (b * s = " << idx_b->value_ * idx_s->value_ << ")";
    const int64_t d = dst_cols->value_;  // feature width (= src cols)
    const int64_t m = dst_rows->value_;
    const DataType dt = input_tile->dtype_;
    // pto.tscatter requires the index element size to match dst (4B→i32, 2/1B→i16).
    const DataType idx_dtype =
        (static_cast<int>(dt.GetBit()) / 8 == 4) ? DataType(DataType::INT32) : DataType(DataType::INT16);
    // For 2-byte dst the tscatter index is i16, so the largest flat destination index
    // (m*d - 1) must fit in i16; otherwise it silently overflows and scatters to wrong rows.
    if (idx_dtype == DataType(DataType::INT16)) {
      CHECK_SPAN(m * d <= 32767, span) << "scatter_update: " << dt.ToString() << " dst has m*d = " << m * d
                                       << " elements, exceeding the i16 flat-index limit (32767); "
                                       << "reduce dst rows*cols or use a 4-byte dtype";
    }
    // Build the flat-index math entirely in i32, then narrow only the finished [n, d] flat_idx
    // to idx_dtype. Every intermediate tile stays in the canonical i32 layout — identical to
    // the FP32 path — which avoids narrowing a small/odd-shaped tile: the i32 [b, s] index has
    // a 32-byte-aligned row (cols * 4), whereas a 2-byte [b, s] tile (cols * 2) does not. The
    // final flat_idx is row-major [n, d] with a 32-byte-aligned row, so casting it to i16 is
    // alignment-legal.
    const DataType compute_dtype(DataType::INT32);

    auto make_idx = [&](int64_t v) -> ExprPtr {
      return std::make_shared<ConstInt>(v, DataType::INDEX, span);
    };
    auto make_cd = [&](int64_t v) -> ExprPtr { return std::make_shared<ConstInt>(v, compute_dtype, span); };
    ExprPtr one = make_idx(1);
    std::vector<std::pair<std::string, std::any>> ci_kw = {{"dtype", compute_dtype}, {"descending", false}};
    std::vector<StmtPtr> prologue;
    auto emit = [&](const std::string& name_op, const std::vector<ExprPtr>& a,
                    const std::vector<std::pair<std::string, std::any>>& kw,
                    const std::string& nm) -> VarPtr {
      auto call = kw.empty() ? op_reg.Create(name_op, a, span) : op_reg.Create(name_op, a, kw, span);
      auto var = std::make_shared<Var>(nm, call->GetType(), span);
      prologue.push_back(std::make_shared<AssignStmt>(var, call, span));
      return var;
    };
    // col_nd[k, c] = c: column arange [1, d] expanded across n rows (tile.ci needs a
    // single-row source, so build [1, d] then col_expand into the [n, d] template).
    auto col_ar = emit("tile.ci", {make_cd(0), MakeShapeTuple({one, make_idx(d)}, span)}, ci_kw, "su_col");
    auto tmpl = emit("tile.full", {MakeShapeTuple({make_idx(n), make_idx(d)}, span), make_cd(0)},
                     {{"dtype", compute_dtype}}, "su_tmpl");
    auto col_nd = emit("tile.col_expand", {tmpl, col_ar}, {}, "su_col_nd");
    // row_base[k] = index.flat[k] * d, broadcast across cols; flat_idx = index.flat[k]*d + c.
    ExprPtr idx_src = args[1];
    if (idx_tile->dtype_ != compute_dtype) {
      idx_src = emit("tile.cast", {idx_src}, {{"target_type", compute_dtype}}, "su_idx_i32");
    }
    auto idx_flat =
        emit("tile.reshape", {idx_src, MakeShapeTuple({make_idx(n), one}, span)}, {}, "su_idx_flat");
    auto row_base = emit("tile.muls", {idx_flat, make_cd(d)}, {}, "su_row_base");
    auto flat_idx = emit("tile.row_expand_add", {col_nd, row_base}, {}, "su_flat_idx");
    // Narrow the finished row-major [n, d] flat indices to the tscatter-required width.
    if (idx_dtype != compute_dtype) {
      flat_idx = emit("tile.cast", {flat_idx}, {{"target_type", idx_dtype}}, "su_flat_idx_cast");
    }

    const int dt_bytes = static_cast<int>(dt.GetBit()) / 8;
    const DataType mask_dt = (dt_bytes == 4)   ? DataType(DataType::FP32)
                             : (dt_bytes == 2) ? DataType(DataType::FP16)
                                               : DataType(DataType::INT8);
    auto make_full = [&](const DataType& f, int64_t r, int64_t c, double v, const std::string& nm) -> VarPtr {
      ExprPtr val = f.IsFloat() ? ExprPtr(std::make_shared<ConstFloat>(v, f, span))
                                : ExprPtr(std::make_shared<ConstInt>(static_cast<int64_t>(v), f, span));
      return emit("tile.full", {MakeShapeTuple({make_idx(r), make_idx(c)}, span), val}, {{"dtype", f}}, nm);
    };
    auto scattered =
        emit("tile.scatter", {make_full(dt, m, d, 0.0, "su_v0"), args[2], flat_idx}, {}, "su_vals");
    auto mask =
        emit("tile.scatter",
             {make_full(mask_dt, m, d, 0.0, "su_m0"), make_full(mask_dt, n, d, 1.0, "su_ones"), flat_idx}, {},
             "su_mask");
    ExprPtr zero_s = mask_dt.IsFloat() ? ExprPtr(std::make_shared<ConstFloat>(0.0, mask_dt, span))
                                       : ExprPtr(std::make_shared<ConstInt>(0, mask_dt, span));
    auto pred = emit("tile.cmps", {mask, zero_s}, {{"cmp_type", 1}}, "su_pred");
    auto tmp = emit("tile.create", {MakeShapeTuple({one, make_idx(32)}, span)},
                    {{"dtype", DataType(DataType::UINT8)}, {"target_memory", MemorySpace::Vec}}, "su_tmp");
    auto out = op_reg.Create("tile.sel", {pred, scattered, args[0], tmp}, span);
    return ConversionResult{std::move(prologue), out};
  };
  // Both the tensor entry and the DSL tile.scatter_update op expand to tile.scatter.
  RegisterCustom("tensor.scatter_update", scatter_update_conv,
                 std::unordered_map<size_t, InputSpaceReq>(scatter_update_input_reqs));
  RegisterCustom("tile.scatter_update", scatter_update_conv, std::move(scatter_update_input_reqs));

  // tensor.create → tile.create with static buffer size validation
  RegisterCustom(
      "tensor.create",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 1, span) << "tensor.create conversion expects 1 arg (shape)";
        auto& op_reg = OpRegistry::GetInstance();

        MemorySpace target_mem = MemorySpace::Vec;
        std::vector<std::pair<std::string, std::any>> new_kwargs;
        for (const auto& [key, value] : kwargs) {
          if (key == "dtype") {
            new_kwargs.emplace_back(key, value);
          }
        }
        new_kwargs.emplace_back("target_memory", target_mem);

        auto shape_tuple = As<MakeTuple>(args[0]);
        DataType dtype = GetKwargOr<DataType>(kwargs, "dtype", DataType::FP32);
        if (shape_tuple && backend::BackendConfig::IsConfigured()) {
          int64_t total_elements = 1;
          bool all_const = true;
          for (const auto& dim : shape_tuple->elements_) {
            if (auto c = As<ConstInt>(dim)) {
              total_elements *= c->value_;
            } else {
              all_const = false;
              break;
            }
          }
          if (all_const) {
            uint64_t tile_bytes = static_cast<uint64_t>(total_elements) * dtype.GetBit() / 8;
            const auto* be = backend::GetBackend();
            if (be) {
              uint64_t mem_size = be->GetMemSize(target_mem);
              INTERNAL_CHECK_SPAN(mem_size == 0 || tile_bytes <= mem_size, span)
                  << "tensor.create: tile size (" << tile_bytes << " bytes) exceeds buffer capacity ("
                  << mem_size << " bytes) for memory space " << static_cast<int>(target_mem) << " at "
                  << span.to_string();
            }
          }
        }

        auto create_call = op_reg.Create("tile.create", args, new_kwargs, span);
        return ConversionResult{create_call};
      });

  // tensor.fillpad → tile.fillpad (with auto-load for TensorType inputs)
  RegisterCustom(
      "tensor.fillpad",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 1, span) << "tensor.fillpad conversion expects 1 arg (input)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        if (As<TileType>(input->GetType())) {
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tile.fillpad", {input}, span)};
          }
          return ConversionResult{op_reg.Create("tile.fillpad", {input}, kwargs, span)};
        }

        auto tensor_type = As<TensorType>(input->GetType());
        INTERNAL_CHECK_SPAN(tensor_type, span)
            << "tensor.fillpad conversion: input must be TensorType or TileType, got "
            << input->GetType()->TypeName();

        auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), span);
        auto shapes = MakeShapeTuple(tensor_type->shape_, span);

        std::vector<ExprPtr> valid_shape = tensor_type->shape_;
        if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->valid_shape.empty()) {
          valid_shape = tensor_type->tensor_view_->valid_shape;
        }
        auto valid_shapes = MakeShapeTuple(valid_shape, span);

        std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                     {"transpose", false}};
        auto load_call =
            op_reg.Create("tile.load", {input, offsets, shapes, valid_shapes}, load_kwargs, span);
        auto load_var = std::make_shared<Var>("fillpad_src", load_call->GetType(), span);

        std::vector<StmtPtr> prologue;
        prologue.push_back(std::make_shared<AssignStmt>(load_var, load_call, span));

        ExprPtr fillpad_call;
        if (kwargs.empty()) {
          fillpad_call = op_reg.Create("tile.fillpad", {load_var}, span);
        } else {
          fillpad_call = op_reg.Create("tile.fillpad", {load_var}, kwargs, span);
        }
        return ConversionResult{std::move(prologue), fillpad_call};
      });

  // tensor.read → tensor.read (gm_tensor) or tile.read (local_tensor)
  // ``AsTensorTypeLike`` matches both ``TensorType`` and ``DistributedTensorType``
  // (a window-bound TensorType subclass). Distributed-tensor reads on the
  // local rank's slice have identical semantics to a plain GM read — they
  // ride the same lowered ``tensor.read`` op.
  RegisterCustom(
      "tensor.read",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 2, span)
            << "tensor.read conversion expects 2 args (tensor, indices)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        if (AsTensorTypeLike(input->GetType())) {
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tensor.read", args, span)};
          }
          return ConversionResult{op_reg.Create("tensor.read", args, kwargs, span)};
        }

        if (As<TileType>(input->GetType())) {
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tile.read", args, span)};
          }
          return ConversionResult{op_reg.Create("tile.read", args, kwargs, span)};
        }

        INTERNAL_UNREACHABLE_SPAN(span)
            << "tensor.read conversion: unexpected input type: " << input->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // tensor.write → tensor.write (gm_tensor) or tile.write (local_tensor)
  // See tensor.read above for the ``AsTensorTypeLike`` rationale.
  RegisterCustom(
      "tensor.write",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 3, span)
            << "tensor.write conversion expects 3 args (tensor, indices, value)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& dest = args[0];

        if (AsTensorTypeLike(dest->GetType())) {
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tensor.write", args, span)};
          }
          return ConversionResult{op_reg.Create("tensor.write", args, kwargs, span)};
        }

        if (As<TileType>(dest->GetType())) {
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tile.write", args, span)};
          }
          return ConversionResult{op_reg.Create("tile.write", args, kwargs, span)};
        }

        INTERNAL_UNREACHABLE_SPAN(span)
            << "tensor.write conversion: unexpected input type: " << dest->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // SPMD block-identity queries: tensor-scope aliases lower 1:1 to tile.* form.
  RegisterSimple("tensor.get_block_idx", "tile.get_block_idx");
  RegisterSimple("tensor.get_subblock_idx", "tile.get_subblock_idx");
  RegisterSimple("tensor.get_block_num", "tile.get_block_num");

  RegisterCustom(
      "tensor.expand_clone",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 2, span)
            << "tensor.expand_clone conversion expects 2 args (input, target)";

        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];
        const auto& target = args[1];

        auto input_tensor_type = As<TensorType>(input->GetType());
        INTERNAL_CHECK_SPAN(input_tensor_type, span)
            << "tensor.expand_clone conversion: input must be TensorType, but got "
            << input->GetType()->TypeName();

        auto target_tensor_type = As<TensorType>(target->GetType());
        INTERNAL_CHECK_SPAN(target_tensor_type, span)
            << "tensor.expand_clone conversion: target must be TensorType, but got "
            << target->GetType()->TypeName();

        const auto& input_shape = input_tensor_type->shape_;
        const auto& target_shape = target_tensor_type->shape_;

        INTERNAL_CHECK_SPAN(input_shape.size() == 3, span)
            << "tensor.expand_clone conversion: input rank must be 3, but got " << input_shape.size();
        INTERNAL_CHECK_SPAN(target_shape.size() == input_shape.size(), span)
            << "tensor.expand_clone conversion: input rank (" << input_shape.size()
            << ") must match target rank (" << target_shape.size() << ")";

        int broadcast_dim = -1;
        for (size_t i = 0; i < input_shape.size(); ++i) {
          if (DimensionsEqual(input_shape[i], target_shape[i])) {
            continue;
          }
          auto input_const = GetConstantDimension(input_shape[i]);
          INTERNAL_CHECK_SPAN(input_const && *input_const == 1, span)
              << "tensor.expand_clone conversion requires input dim " << i
              << " to be 1 for broadcasting, but got " << PythonPrint(input_shape[i]);
          INTERNAL_CHECK_SPAN(broadcast_dim < 0, span)
              << "tensor.expand_clone conversion allows broadcasting in at most one dimension";
          broadcast_dim = static_cast<int>(i);
        }

        std::vector<StmtPtr> prologue;

        auto make_index_const = [&](int64_t value) -> ExprPtr {
          return std::make_shared<ConstInt>(value, DataType::INDEX, span);
        };

        auto make_tuple = [&](std::vector<ExprPtr> elems) -> ExprPtr {
          return std::make_shared<MakeTuple>(std::move(elems), span);
        };

        auto load_tensor_tile = [&](const ExprPtr& tensor, const ExprPtr& offsets,
                                    const std::vector<ExprPtr>& shape,
                                    const std::vector<ExprPtr>& valid_shape, const std::string& name_hint,
                                    std::vector<StmtPtr>& stmts) -> ExprPtr {
          auto shapes = MakeShapeTuple(shape, span);
          auto valid_shapes = MakeShapeTuple(valid_shape, span);
          std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                       {"transpose", false}};
          auto load_call =
              op_reg.Create("tile.load", {tensor, offsets, shapes, valid_shapes}, load_kwargs, span);
          auto load_var = std::make_shared<Var>(name_hint, load_call->GetType(), span);
          stmts.push_back(std::make_shared<AssignStmt>(load_var, load_call, span));
          return load_var;
        };

        DataType input_dtype = input_tensor_type->dtype_;

        std::vector<ExprPtr> input_valid_shape = input_shape;
        if (input_tensor_type && input_tensor_type->tensor_view_.has_value() &&
            !input_tensor_type->tensor_view_->valid_shape.empty()) {
          input_valid_shape = input_tensor_type->tensor_view_->valid_shape;
        }

        ExprPtr zero = make_index_const(0);
        ExprPtr one = make_index_const(1);

        if (broadcast_dim < 0) {
          ExprPtr input_tile = input;
          auto offsets = MakeZeroOffsets(input_shape.size(), span);
          input_tile = load_tensor_tile(input, offsets, input_shape, input_valid_shape, "expand_clone_input",
                                        prologue);
          auto store_call = op_reg.Create("tile.store", {input_tile, offsets, target}, span);
          return ConversionResult{std::move(prologue), store_call};
        }

        if (broadcast_dim == 0) {
          ExprPtr input_tile = input;
          auto offsets = MakeZeroOffsets(input_tensor_type->shape_.size(), span);
          input_tile = load_tensor_tile(input, offsets, input_shape, input_valid_shape, "expand_clone_input",
                                        prologue);

          auto loop_var = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INDEX), span);
          auto iter_arg = std::make_shared<IterArg>("expand_clone_acc", target_tensor_type, target, span);
          auto return_var = std::make_shared<Var>("expand_clone_d0_result", target_tensor_type, span);

          auto loop_offsets = make_tuple({loop_var, zero, zero});
          auto store_call = op_reg.Create("tile.store", {input_tile, loop_offsets, iter_arg}, span);
          auto store_var = std::make_shared<Var>("expand_clone_d0_store", store_call->GetType(), span);

          std::vector<StmtPtr> body_stmts;
          body_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, span));
          body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{store_var}, span));

          auto body = SeqStmts::Flatten(std::move(body_stmts), span);
          auto for_stmt = std::make_shared<ForStmt>(loop_var, zero, target_shape[0], one,
                                                    std::vector<IterArgPtr>{iter_arg}, body,
                                                    std::vector<VarPtr>{return_var}, span);
          prologue.push_back(for_stmt);
          return ConversionResult{std::move(prologue), return_var};
        }

        if (broadcast_dim == 1) {
          auto loop_var = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INDEX), span);
          auto iter_arg = std::make_shared<IterArg>("expand_clone_acc", target_tensor_type, target, span);
          auto return_var = std::make_shared<Var>("expand_clone_d1_result", target_tensor_type, span);

          auto loop_offsets = make_tuple({loop_var, zero, zero});
          std::vector<ExprPtr> slice_shape = {one, one, input_valid_shape[2]};

          std::vector<StmtPtr> body_stmts;
          auto input_tile = load_tensor_tile(input, loop_offsets, slice_shape, slice_shape,
                                             "expand_clone_d1_input", body_stmts);

          std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", input_dtype},
                                                                         {"target_memory", MemorySpace::Vec}};
          auto create_shape = MakeShapeTuple({one, target_shape[1], target_shape[2]}, span);
          auto create_call = op_reg.Create("tile.create", {create_shape}, create_kwargs, span);
          auto create_var = std::make_shared<Var>("expand_clone_d1_target", create_call->GetType(), span);
          body_stmts.push_back(std::make_shared<AssignStmt>(create_var, create_call, span));

          auto col_expand_call = op_reg.Create("tile.col_expand", {create_var, input_tile}, span);
          auto col_expand_var =
              std::make_shared<Var>("expand_clone_d1_col", col_expand_call->GetType(), span);
          body_stmts.push_back(std::make_shared<AssignStmt>(col_expand_var, col_expand_call, span));

          auto store_call = op_reg.Create("tile.store", {col_expand_var, loop_offsets, iter_arg}, span);
          auto store_var = std::make_shared<Var>("expand_clone_d1_store", store_call->GetType(), span);
          body_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, span));
          body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{store_var}, span));

          auto body = SeqStmts::Flatten(std::move(body_stmts), span);
          auto for_stmt = std::make_shared<ForStmt>(loop_var, zero, target_shape[0], one,
                                                    std::vector<IterArgPtr>{iter_arg}, body,
                                                    std::vector<VarPtr>{return_var}, span);
          prologue.push_back(for_stmt);
          return ConversionResult{std::move(prologue), return_var};
        }

        auto offsets = MakeZeroOffsets(target_shape.size(), span);
        auto input_tile =
            load_tensor_tile(input, offsets, input_shape, input_valid_shape, "expand_clone_input", prologue);

        std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", input_dtype},
                                                                       {"target_memory", MemorySpace::Vec}};
        auto create_shape = MakeShapeTuple(target_shape, span);
        auto create_call = op_reg.Create("tile.create", {create_shape}, create_kwargs, span);
        auto create_var = std::make_shared<Var>("expand_clone_d2_target", create_call->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(create_var, create_call, span));

        auto row_expand_call = op_reg.Create("tile.row_expand", {create_var, input_tile}, span);
        auto row_expand_var = std::make_shared<Var>("expand_clone_d2_row", row_expand_call->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(row_expand_var, row_expand_call, span));
        auto store_call = op_reg.Create("tile.store", {row_expand_var, offsets, target}, span);
        return ConversionResult{std::move(prologue), store_call};
      });
}

// ============================================================================
// Matmul ops: tensor.matmul / tensor.matmul_acc with Mat-space input_reqs
// ============================================================================

void OpConversionRegistry::RegisterMatmulOps() {
  // Helper: report rank of a Tensor or Tile typed argument. By the time the
  // conversion lambda runs, BridgeInputSpaces has already loaded TensorType args
  // into TileType operands per input_reqs, so the args we see here may be either
  // tile- or tensor-typed.
  auto rank_of = [](const ExprPtr& e) -> size_t {
    if (auto t = As<TileType>(e->GetType())) return t->shape_.size();
    if (auto t = As<TensorType>(e->GetType())) return t->shape_.size();
    INTERNAL_UNREACHABLE << "matmul conversion: argument has unexpected type " << e->GetType()->TypeName();
  };

  // tensor.matmul: 2D × 2D → tile.matmul; any operand ≥3D → tile.batch_matmul.
  // a_trans/b_trans are honored via InputSpaceReq below — the producer load is
  // emitted with target_memory=Mat and transpose=True, so the transposed tile
  // arrives at matmul/batch_matmul already in the correct orientation.
  RegisterCustom(
      "tensor.matmul",
      [rank_of](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
                const Span& span) -> ConversionResult {
        (void)kwargs;
        INTERNAL_CHECK_SPAN(args.size() == 2, span) << "tensor.matmul conversion expects 2 args (lhs, rhs)";
        const bool nd = rank_of(args[0]) > 2 || rank_of(args[1]) > 2;
        const std::string out_op = nd ? "tile.batch_matmul" : "tile.matmul";
        return ConversionResult{OpRegistry::GetInstance().Create(out_op, {args[0], args[1]}, span)};
      },
      {{0, {MemorySpace::Mat, "a_trans"}}, {1, {MemorySpace::Mat, "b_trans"}}});

  // tensor.matmul_acc: 2D × 2D × 2D → tile.matmul_acc; any operand ≥3D →
  // tile.batch_matmul_acc. Same a_trans/b_trans handling as tensor.matmul.
  RegisterCustom(
      "tensor.matmul_acc",
      [rank_of](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
                const Span& span) -> ConversionResult {
        (void)kwargs;
        INTERNAL_CHECK_SPAN(args.size() == 3, span)
            << "tensor.matmul_acc conversion expects 3 args (acc, lhs, rhs)";
        const bool nd = rank_of(args[0]) > 2 || rank_of(args[1]) > 2 || rank_of(args[2]) > 2;
        const std::string out_op = nd ? "tile.batch_matmul_acc" : "tile.matmul_acc";
        return ConversionResult{OpRegistry::GetInstance().Create(out_op, {args[0], args[1], args[2]}, span)};
      },
      {{1, {MemorySpace::Mat, "a_trans"}}, {2, {MemorySpace::Mat, "b_trans"}}});
}

// ============================================================================
// Reduction ops: row_max, row_sum, row_min (with tmp_tile workspace)
// ============================================================================

void OpConversionRegistry::RegisterReductionOps() {
  auto MakeReductionConv = [](const std::string& tile_op) -> ConversionFunc {
    return [tile_op](const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                     const Span& span) -> ConversionResult {
      INTERNAL_CHECK_SPAN(args.size() == 1, span) << tile_op << " conversion expects 1 arg (input tile)";
      auto& op_reg = OpRegistry::GetInstance();

      const auto& input = args[0];
      auto tile_type = As<TileType>(input->GetType());
      INTERNAL_CHECK_SPAN(tile_type, span)
          << tile_op << " conversion: input must be TileType, got " << input->GetType()->TypeName();

      std::vector<ExprPtr> tmp_shape = tile_type->shape_;
      if (tmp_shape.size() >= 2) {
        auto last = As<ConstInt>(tmp_shape.back());
        if (!last || last->value_ < 128) {
          tmp_shape.back() = std::make_shared<ConstInt>(128, DataType::INDEX, span);
        }
      }
      auto shape_tuple = std::make_shared<MakeTuple>(tmp_shape, span);
      std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", tile_type->dtype_},
                                                                     {"target_memory", MemorySpace::Vec}};
      auto create_call = op_reg.Create("tile.create", {shape_tuple}, create_kwargs, span);

      auto tmp_var = std::make_shared<Var>("tmp_tile", create_call->GetType(), span);
      std::vector<StmtPtr> prologue;
      prologue.push_back(std::make_shared<AssignStmt>(tmp_var, create_call, span));

      auto reduction_call = op_reg.Create(tile_op, {input, tmp_var}, span);
      return ConversionResult{std::move(prologue), reduction_call};
    };
  };

  RegisterCustom("tensor.row_max", MakeReductionConv("tile.row_max"));
  RegisterCustom("tensor.row_sum", MakeReductionConv("tile.row_sum"));
  RegisterCustom("tensor.row_min", MakeReductionConv("tile.row_min"));
  RegisterCustom("tensor.row_prod", MakeReductionConv("tile.row_prod"));

  // tile.col_sum's 1-arg form is the sequential reduction path — no tmp_tile workspace
  // needed, so a plain 1:1 name rewrite is enough. tile.col_max / tile.col_min are
  // likewise 1-arg, so the same simple rewrite applies.
  RegisterSimple("tensor.col_sum", "tile.col_sum");
  RegisterSimple("tensor.col_max", "tile.col_max");
  RegisterSimple("tensor.col_min", "tile.col_min");
  RegisterSimple("tensor.col_prod", "tile.col_prod");
}

// ============================================================================
// Sort ops: sort32, mrgsort_format1, mrgsort_format2 — simple 1:1 name mapping.
// Auto-bridge in the convert pass loads TensorType args to Vec tiles.
// ============================================================================

void OpConversionRegistry::RegisterSortOps() {
  RegisterSimple("tensor.sort32", "tile.sort32");
  RegisterSimple("tensor.mrgsort_format1", "tile.mrgsort_format1");
  RegisterSimple("tensor.gather_mask", "tile.gather_mask");

  // tensor.mrgsort_format2: 2-4 srcs → tile.mrgsort_format2 with a synthesized
  // scratch tmp tile allocated locally in Vec memory.
  //
  // The tile-level op requires (srcs..., tmp). We don't expose tmp at the
  // tensor level because its shape equals the merged output shape (sum of src
  // last dims) — we can derive it; there is no user-visible value. The per-way
  // "executed" status is a vector<4xi16> output of pto.tmrgsort that codegen
  // synthesizes directly, so no executed tile is plumbed through the IR.
  //
  // Inputs (srcs) are auto-bridged to Vec tiles by the framework (input_reqs).
  std::unordered_map<size_t, InputSpaceReq> mrgsort2_input_reqs;
  for (size_t i = 0; i < 4; ++i) {
    mrgsort2_input_reqs[i] = {MemorySpace::Vec, std::nullopt};
  }
  RegisterCustom(
      "tensor.mrgsort_format2",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() >= 2 && args.size() <= 4, span)
            << "tensor.mrgsort_format2 conversion expects 2-4 src args, got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();

        // After the framework's input_reqs bridge, all srcs should be Vec tiles.
        std::vector<std::shared_ptr<const TileType>> src_tile_types;
        src_tile_types.reserve(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
          auto tt = As<TileType>(args[i]->GetType());
          INTERNAL_CHECK_SPAN(tt, span)
              << "tensor.mrgsort_format2 conversion expects bridged Vec tile at arg " << i;
          src_tile_types.push_back(tt);
        }
        const auto& src0_tile = src_tile_types.front();

        // tmp shape = same rank as src0, last dim = sum of all srcs' last dims.
        std::vector<ExprPtr> tmp_shape(src0_tile->shape_.begin(), src0_tile->shape_.end() - 1);
        int64_t const_sum = 0;
        bool all_const = true;
        for (const auto& st : src_tile_types) {
          auto c = As<ConstInt>(st->shape_.back());
          if (!c) {
            all_const = false;
            break;
          }
          const_sum += c->value_;
        }
        ExprPtr last_dim;
        if (all_const) {
          last_dim = std::make_shared<ConstInt>(const_sum, DataType::INDEX, span);
        } else {
          last_dim = src_tile_types[0]->shape_.back();
          for (size_t i = 1; i < src_tile_types.size(); ++i) {
            last_dim =
                std::make_shared<Add>(last_dim, src_tile_types[i]->shape_.back(), DataType::INDEX, span);
          }
        }
        tmp_shape.push_back(last_dim);

        std::vector<StmtPtr> prologue;

        // Synthesize tmp: tile.create(tmp_shape, dtype=src0.dtype, target_memory=Vec)
        auto tmp_shape_tuple = std::make_shared<MakeTuple>(tmp_shape, span);
        std::vector<std::pair<std::string, std::any>> tmp_create_kwargs = {
            {"dtype", src0_tile->dtype_}, {"target_memory", MemorySpace::Vec}};
        auto tmp_create = op_reg.Create("tile.create", {tmp_shape_tuple}, tmp_create_kwargs, span);
        auto tmp_var = std::make_shared<Var>("mrgsort2_tmp", tmp_create->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(tmp_var, tmp_create, span));

        // Assemble tile.mrgsort_format2 call: (src0..srcN-1, tmp) + kwargs.
        std::vector<ExprPtr> tile_args(args.begin(), args.end());
        tile_args.push_back(tmp_var);
        auto mrgsort_call = op_reg.Create("tile.mrgsort_format2", tile_args, kwargs, span);
        return ConversionResult{std::move(prologue), mrgsort_call};
      },
      std::move(mrgsort2_input_reqs));
}

// ============================================================================
// Generalized gather lowering.
//
// Hardware constraint: pto.tgather only works correctly when the source tile
// has exactly 1 row (rows=1).  Therefore all lowering paths use ForStmt loops
// to decompose the gather into single-row pto.tgather calls.
//
// FlattenTileNdTo2D constraint: tile.load, tile.store, tile.reshape may
// produce/consume >2D tiles; all other tile ops must be 2D.
// tile.load with an N-D shape is automatically flattened to 2D by merging
// all leading dims: [d0,...,d_{n-1}] → [d0*…*d_{n-2}, d_{n-1}].
// Because of this, we explicitly tile.reshape every N-D tile.load result to
// 2D before passing it to any other op.
//
// Storage for rank-3 output: we return a 2D tile [I0*I1, I2] (where I2 is
// the tensor's last dim).  FlattenTileNdTo2D injects partition_shape
// [1, I0*I1, I2] for the resulting tile.store, so element [0,j,k] maps to
// physical j*I2+k — covering all I0*I1*I2 elements without overlap.
// We always add a trailing tile.reshape so Phase 3 (RewriteReturnedAssemble-
// LoopToStore) does not fire; we want the full-tile store path instead.
//
// Four cases (by rank and norm_dim):
//
// Case 1  rank==2, dim==1 (last):
//   Loop over I0 rows: load [1,S1] and [1,K], single-row gather.
//   Accumulator [I0, K].  Phase 3 rewrites the loop to per-row tile.store.
//
// Case 2  rank==3, dim==2 (last):
//   Nested loop: outer I0 × inner I1.
//   Load [1,1,S2]→reshape[1,S2]; Load [1,1,K]→reshape[1,K]; gather [1,K].
//   Inner acc [I1,K]; reshape→[1,I1*K]; outer acc [I0,I1*K].
//   Final reshape [I0,I1*K]→[I0*I1,K]; tile.store at [0,0,0].
//
// Case 3  rank==3, dim==0 (first):
//   Flat-index gather: for each output row r = i0*I1+i1:
//     inp_flat = inp[:, i1, :].flatten()  → [1, S0*I2]
//     idx_row  = idx[i0, i1, :]           → [1, I2]
//     flat_idx = idx_row * I2 + [0..I2-1] → [1, I2]
//     out_row  = gather(inp_flat, flat_idx) → [1, I2]
//   Accumulator [I0*I1, I2]; reshape→[I0*I1,I2]; tile.store at [0,0,0].
//
// Case 4  rank==3, dim==1 (middle):
//   Flat-index gather: for each output row r = i0*I1+i1:
//     inp_flat = inp[i0, :, :].flatten()  → [1, S1*I2]
//     idx_row  = idx[i0, i1, :]           → [1, I2]
//     flat_idx = idx_row * I2 + [0..I2-1] → [1, I2]
//     out_row  = gather(inp_flat, flat_idx) → [1, I2]
//   Accumulator [I0*I1, I2]; reshape→[I0*I1,I2]; tile.store at [0,0,0].
// ============================================================================

void OpConversionRegistry::RegisterGatherOps() {
  // tensor.gather (index form) — accepts TensorType or TileType for input/index.
  // Upstream conversions may already have lowered an argument to a tile, so we
  // emit tile.load when the source is still a tensor and tile.slice when it is
  // already a tile (the latter only safe for 2D; FlattenTileNdTo2D rejects >2D
  // tile.slice).
  RegisterCustom(
      "tensor.gather",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 2, span)
            << "tensor.gather conversion expects 2 args (input, index), got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();

        const auto& input = args[0];
        const auto& index = args[1];

        auto get_shape_dtype = [&](const ExprPtr& e,
                                   const char* role) -> std::pair<std::vector<ExprPtr>, DataType> {
          if (auto t = As<TensorType>(e->GetType())) return {t->shape_, t->dtype_};
          if (auto t = As<TileType>(e->GetType())) return {t->shape_, t->dtype_};
          INTERNAL_UNREACHABLE_SPAN(span)
              << "tensor.gather conversion: " << role << " must be TensorType or TileType, got "
              << e->GetType()->TypeName();
          return {};  // unreachable
        };
        auto input_info = get_shape_dtype(input, "input");
        auto index_info = get_shape_dtype(index, "index");
        const auto& input_shape = input_info.first;
        const DataType input_dtype = input_info.second;
        const auto& index_shape = index_info.first;
        const int64_t rank = static_cast<int64_t>(input_shape.size());
        INTERNAL_CHECK_SPAN(rank >= 2, span) << "tensor.gather conversion: rank must be >= 2, got " << rank;

        int dim_val = GetKwargOr<int>(kwargs, "dim", -1);
        int norm_dim = dim_val < 0 ? dim_val + static_cast<int>(rank) : dim_val;
        INTERNAL_CHECK_SPAN(norm_dim >= 0 && norm_dim < static_cast<int>(rank), span)
            << "tensor.gather conversion: dim out of range, got " << dim_val;

        auto make_idx = [&](int64_t value) -> ExprPtr {
          return std::make_shared<ConstInt>(value, DataType::INDEX, span);
        };
        auto make_i32 = [&](int64_t value) -> ExprPtr {
          return std::make_shared<ConstInt>(value, DataType::INT32, span);
        };
        auto zero = make_idx(0);
        auto one = make_idx(1);

        std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                     {"transpose", false}};
        std::vector<std::pair<std::string, std::any>> tmp_create_kwargs = {
            {"dtype", DataType(DataType::INT32)}, {"target_memory", MemorySpace::Vec}};

        std::vector<StmtPtr> prologue;

        // --- Low-level helpers ---

        auto emit_to = [&](std::vector<StmtPtr>& stmts, const std::string& op_name,
                           const std::vector<ExprPtr>& op_args,
                           const std::vector<std::pair<std::string, std::any>>& op_kwargs,
                           const std::string& name) -> VarPtr {
          auto call = op_kwargs.empty() ? op_reg.Create(op_name, op_args, span)
                                        : op_reg.Create(op_name, op_args, op_kwargs, span);
          auto var = std::make_shared<Var>(name, call->GetType(), span);
          stmts.push_back(std::make_shared<AssignStmt>(var, call, span));
          return var;
        };

        auto emit = [&](const std::string& op_name, const std::vector<ExprPtr>& op_args,
                        const std::vector<std::pair<std::string, std::any>>& op_kwargs,
                        const std::string& name) -> VarPtr {
          return emit_to(prologue, op_name, op_args, op_kwargs, name);
        };

        // Emit tile.reshape.
        auto reshape_to = [&](std::vector<StmtPtr>& stmts, const ExprPtr& src,
                              const std::vector<ExprPtr>& new_shape, const std::string& name) -> VarPtr {
          return emit_to(stmts, "tile.reshape", {src, MakeShapeTuple(new_shape, span)}, {}, name);
        };

        auto emit_load_or_slice = [&](std::vector<StmtPtr>& stmts, const ExprPtr& src, const ExprPtr& offsets,
                                      const ExprPtr& shape, const std::string& name) -> VarPtr {
          if (As<TileType>(src->GetType())) {
            return emit_to(stmts, "tile.slice", {src, shape, offsets, shape}, {}, name);
          }
          return emit_to(stmts, "tile.load", {src, offsets, shape, shape}, load_kwargs, name);
        };

        // Emit single-row tile.gather (with scratch tile); src_row and idx_row must be 2D.
        auto single_row_gather = [&](std::vector<StmtPtr>& stmts, const VarPtr& src_row,
                                     const VarPtr& idx_row, int64_t idx_cols,
                                     const std::string& name) -> VarPtr {
          auto tmp_sh = MakeShapeTuple({one, make_idx(idx_cols)}, span);
          auto tmp = emit_to(stmts, "tile.create", {tmp_sh}, tmp_create_kwargs, name + "_tmp");
          return emit_to(stmts, "tile.gather", {src_row, idx_row, tmp}, {}, name);
        };

        // Build a ForStmt that accumulates [1, acc_cols] rows into [acc_rows, acc_cols].
        // body_builder receives (loop_var, iter_arg, body_stmts) and returns a [1, acc_cols] tile.
        auto make_loop =
            [&](std::vector<StmtPtr>& outer_stmts, const std::string& lname, const ExprPtr& loop_stop,
                int64_t acc_rows, int64_t acc_cols, DataType acc_dtype,
                const std::function<VarPtr(const VarPtr&, const IterArgPtr&, std::vector<StmtPtr>&)>&
                    body_builder) -> VarPtr {
          std::vector<std::pair<std::string, std::any>> acc_kwargs = {{"dtype", acc_dtype},
                                                                      {"target_memory", MemorySpace::Vec}};
          auto acc_init = emit_to(outer_stmts, "tile.create",
                                  {MakeShapeTuple({make_idx(acc_rows), make_idx(acc_cols)}, span)},
                                  acc_kwargs, lname + "_acc_init");
          auto acc_type = acc_init->GetType();
          auto lv = std::make_shared<Var>(lname + "_lv", std::make_shared<ScalarType>(DataType::INDEX), span);
          auto ia = std::make_shared<IterArg>(lname + "_ia", acc_type, acc_init, span);
          auto rv = std::make_shared<Var>(lname + "_rv", acc_type, span);

          std::vector<StmtPtr> body_stmts;
          auto row_result = body_builder(lv, ia, body_stmts);
          auto ofs = std::make_shared<MakeTuple>(std::vector<ExprPtr>{lv, zero}, span);
          auto asmbl = emit_to(body_stmts, "tile.assemble", {ia, row_result, ofs}, {}, lname + "_asmbl");
          body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{asmbl}, span));
          auto body = SeqStmts::Flatten(std::move(body_stmts), span);
          outer_stmts.push_back(std::make_shared<ForStmt>(
              lv, zero, loop_stop, one, std::vector<IterArgPtr>{ia}, body, std::vector<VarPtr>{rv}, span));
          return rv;
        };

        // Get ConstInt value from a shape expression.
        auto get_const = [&](const ExprPtr& expr, const char* what) -> int64_t {
          auto c = As<ConstInt>(expr);
          INTERNAL_CHECK_SPAN(c, span)
              << "tensor.gather: " << what << " must be ConstInt for rank>2 lowering";
          return c->value_;
        };

        // ================================================================
        // Case 1  rank==2, dim==1 (last dim)
        // ================================================================
        if (rank == 2 && norm_dim == 1) {
          int64_t I0 = get_const(index_shape[0], "index.shape[0]");
          int64_t S1 = get_const(input_shape[1], "input.shape[1]");
          int64_t K = get_const(index_shape[1], "index.shape[1]");

          auto result =
              make_loop(prologue, "gather", index_shape[0], I0, K, input_dtype,
                        [&](const VarPtr& lv, const IterArgPtr& /*ia*/, std::vector<StmtPtr>& bs) -> VarPtr {
                          auto row_ofs = std::make_shared<MakeTuple>(std::vector<ExprPtr>{lv, zero}, span);
                          auto inp_sh = MakeShapeTuple({one, make_idx(S1)}, span);
                          auto inp_row = emit_load_or_slice(bs, input, row_ofs, inp_sh, "gather_inp_row");
                          auto idx_sh = MakeShapeTuple({one, make_idx(K)}, span);
                          auto idx_row = emit_load_or_slice(bs, index, row_ofs, idx_sh, "gather_idx_row");
                          return single_row_gather(bs, inp_row, idx_row, K, "gather_row");
                        });
          return ConversionResult{std::move(prologue), result};
        }

        // ================================================================
        // Case 2  rank==3, dim==2 (last dim)
        // Result tile: [I0*I1, K] where tile[i0*I1+i1, k] = output[i0, i1, k].
        // Stored via tile.store at [0,0,0]; FlattenTileNdTo2D injects
        // partition_shape [1, I0*I1, K] which covers all elements correctly.
        // ================================================================
        if (rank == 3 && norm_dim == 2) {
          int64_t I0 = get_const(index_shape[0], "index.shape[0]");
          int64_t I1 = get_const(index_shape[1], "index.shape[1]");
          int64_t S2 = get_const(input_shape[2], "input.shape[2]");
          int64_t K = get_const(index_shape[2], "index.shape[2]");
          int64_t I1K = I1 * K;

          // Outer loop: i0=0..I0-1, accumulates [I0, I1*K].
          auto outer_result = make_loop(
              prologue, "gather_outer", index_shape[0], I0, I1K, input_dtype,
              [&](const VarPtr& outer_lv, const IterArgPtr& /*oia*/, std::vector<StmtPtr>& ob) -> VarPtr {
                // Inner loop: i1=0..I1-1, accumulates [I1, K].
                auto inner_result =
                    make_loop(ob, "gather_inner", index_shape[1], I1, K, input_dtype,
                              [&](const VarPtr& inner_lv, const IterArgPtr& /*iia*/,
                                  std::vector<StmtPtr>& bs) -> VarPtr {
                                auto ofs = std::make_shared<MakeTuple>(
                                    std::vector<ExprPtr>{outer_lv, inner_lv, zero}, span);
                                // Load with 3D shape → 3D tile type; immediately reshape to 2D.
                                auto inp_sh = MakeShapeTuple({one, one, make_idx(S2)}, span);
                                auto inp_raw = emit_load_or_slice(bs, input, ofs, inp_sh, "gather_inp_raw");
                                auto inp_row = reshape_to(bs, inp_raw, {one, make_idx(S2)}, "gather_inp_row");
                                auto idx_sh = MakeShapeTuple({one, one, make_idx(K)}, span);
                                auto idx_raw = emit_load_or_slice(bs, index, ofs, idx_sh, "gather_idx_raw");
                                auto idx_row = reshape_to(bs, idx_raw, {one, make_idx(K)}, "gather_idx_row");
                                return single_row_gather(bs, inp_row, idx_row, K, "gather_row");
                              });
                // Reshape [I1, K] → [1, I1*K] for outer assemble.
                return reshape_to(ob, inner_result, {one, make_idx(I1K)}, "gather_inner_flat");
              });
          // Reshape [I0, I1*K] → [I0*I1, K].  Prevents Phase 3 and gives correct 2D layout.
          int64_t I0I1 = I0 * I1;
          auto out_2d = reshape_to(prologue, outer_result, {make_idx(I0I1), make_idx(K)}, "gather_out");
          return ConversionResult{std::move(prologue), out_2d};
        }

        // ================================================================
        // Case 3  rank==3, dim==0 (first dim)
        // out[i0, i1, k] = inp[idx[i0, i1, k], i1, k]
        // Result tile: [I0*I1, I2] where tile[i0*I1+i1, k] = output[i0, i1, k].
        //
        // Uses flat-index gather to avoid intermediate tiles with I0 (potentially
        // non-8-aligned) columns, which would violate hardware 32-byte row alignment.
        // For each output row r = i0*I1+i1:
        //   inp_flat = inp[:, i1, :].flatten()  → [1, S0*S2]
        //   idx_row  = idx[i0, i1, :]           → [1, I2]
        //   flat_idx = idx_row * S2 + [0..I2-1] → [1, I2]
        //   out_row  = gather(inp_flat, flat_idx) → [1, I2]
        // ================================================================
        if (rank == 3 && norm_dim == 0) {
          int64_t S0 = get_const(input_shape[0], "input.shape[0]");
          int64_t S2 = get_const(input_shape[2], "input.shape[2]");
          int64_t I0 = get_const(index_shape[0], "index.shape[0]");
          int64_t I1 = get_const(index_shape[1], "index.shape[1]");
          int64_t I2 = get_const(index_shape[2], "index.shape[2]");
          int64_t I0I1 = I0 * I1;
          int64_t S0S2 = S0 * S2;

          // Precompute constant range tile [0, 1, ..., I2-1] (shared across all loop iterations).
          std::vector<std::pair<std::string, std::any>> ci_kw = {{"dtype", DataType(DataType::INT32)}};
          auto range_1d = emit("tile.ci", {make_i32(0), MakeShapeTuple({one, make_idx(I2)}, span)}, ci_kw,
                               "gather_range");

          // Outer loop: r=0..I0*I1-1, accumulating [I0*I1, I2].
          auto result = make_loop(
              prologue, "gather_main", make_idx(I0I1), I0I1, I2, input_dtype,
              [&](const VarPtr& lv, const IterArgPtr& /*ia*/, std::vector<StmtPtr>& bs) -> VarPtr {
                auto i0_expr = MakeFloorDiv(lv, make_idx(I1), span);
                auto i1_expr = MakeFloorMod(lv, make_idx(I1), span);

                // Load inp[:, i1, :] → [S0, 1, I2] → [S0, I2] → [1, S0*I2].
                auto inp_ofs = std::make_shared<MakeTuple>(std::vector<ExprPtr>{zero, i1_expr, zero}, span);
                auto inp_sh = MakeShapeTuple({input_shape[0], one, input_shape[2]}, span);
                auto inp_raw = emit_load_or_slice(bs, input, inp_ofs, inp_sh, "gather_inp_raw");
                auto inp_2d = reshape_to(bs, inp_raw, {input_shape[0], input_shape[2]}, "gather_inp_2d");
                auto inp_flat = reshape_to(bs, inp_2d, {one, make_idx(S0S2)}, "gather_inp_flat");

                // Load idx[i0, i1, :] → [1, 1, I2] → [1, I2].
                auto idx_ofs =
                    std::make_shared<MakeTuple>(std::vector<ExprPtr>{i0_expr, i1_expr, zero}, span);
                auto idx_sh = MakeShapeTuple({one, one, index_shape[2]}, span);
                auto idx_raw = emit_load_or_slice(bs, index, idx_ofs, idx_sh, "gather_idx_raw");
                auto idx_row = reshape_to(bs, idx_raw, {one, index_shape[2]}, "gather_idx_row");

                // flat_idx[k] = idx_row[k] * S2 + k  →  selects inp_flat[flat_idx[k]].
                auto idx_sc = emit_to(bs, "tile.muls", {idx_row, make_i32(S2)}, {}, "gather_idx_s");
                auto flat_idx = emit_to(bs, "tile.add", {idx_sc, range_1d}, {}, "gather_fidx");

                return single_row_gather(bs, inp_flat, flat_idx, I2, "gather_row");
              });
          // Reshape [I0*I1, I2] is already the correct 2D layout; prevents Phase 3 optimization.
          auto out_2d = reshape_to(prologue, result, {make_idx(I0I1), make_idx(I2)}, "gather_out");
          return ConversionResult{std::move(prologue), out_2d};
        }

        // ================================================================
        // Case 4  rank==3, dim==1 (middle dim)
        // out[i0, i1, k] = inp[i0, idx[i0, i1, k], k]
        // Result tile: [I0*I1, I2] where tile[i0*I1+i1, k] = output[i0, i1, k].
        //
        // Uses flat-index gather to avoid intermediate tiles with I1 (potentially
        // non-8-aligned) columns, which would violate hardware 32-byte row alignment.
        // For each output row r = i0*I1+i1:
        //   inp_flat = inp[i0, :, :].flatten()  → [1, S1*S2]
        //   idx_row  = idx[i0, i1, :]           → [1, I2]
        //   flat_idx = idx_row * S2 + [0..I2-1] → [1, I2]
        //   out_row  = gather(inp_flat, flat_idx) → [1, I2]
        // ================================================================
        CHECK_SPAN(rank == 3 && norm_dim == 1, span) << "tensor.gather: unsupported (rank, dim) combination, "
                                                     << "got rank=" << rank << " norm_dim=" << norm_dim;

        {
          int64_t I0 = get_const(index_shape[0], "index.shape[0]");
          int64_t I1 = get_const(index_shape[1], "index.shape[1]");
          int64_t I2 = get_const(index_shape[2], "index.shape[2]");
          int64_t S1 = get_const(input_shape[1], "input.shape[1]");
          int64_t S2 = get_const(input_shape[2], "input.shape[2]");
          int64_t I0I1 = I0 * I1;
          int64_t S1S2 = S1 * S2;

          // Precompute constant range tile [0, 1, ..., I2-1] (shared across all loop iterations).
          std::vector<std::pair<std::string, std::any>> ci_kw = {{"dtype", DataType(DataType::INT32)}};
          auto range_1d = emit("tile.ci", {make_i32(0), MakeShapeTuple({one, make_idx(I2)}, span)}, ci_kw,
                               "gather_range");

          // Outer loop: r=0..I0*I1-1, accumulating [I0*I1, I2].
          auto result = make_loop(
              prologue, "gather_main", make_idx(I0I1), I0I1, I2, input_dtype,
              [&](const VarPtr& lv, const IterArgPtr& /*ia*/, std::vector<StmtPtr>& bs) -> VarPtr {
                auto i0_expr = MakeFloorDiv(lv, make_idx(I1), span);
                auto i1_expr = MakeFloorMod(lv, make_idx(I1), span);

                // Load inp[i0, :, :] → [1, S1, I2] → [S1, I2] → [1, S1*I2].
                auto inp_ofs = std::make_shared<MakeTuple>(std::vector<ExprPtr>{i0_expr, zero, zero}, span);
                auto inp_sh = MakeShapeTuple({one, input_shape[1], input_shape[2]}, span);
                auto inp_raw = emit_load_or_slice(bs, input, inp_ofs, inp_sh, "gather_inp_raw");
                auto inp_2d = reshape_to(bs, inp_raw, {input_shape[1], input_shape[2]}, "gather_inp_2d");
                auto inp_flat = reshape_to(bs, inp_2d, {one, make_idx(S1S2)}, "gather_inp_flat");

                // Load idx[i0, i1, :] → [1, 1, I2] → [1, I2].
                auto idx_ofs =
                    std::make_shared<MakeTuple>(std::vector<ExprPtr>{i0_expr, i1_expr, zero}, span);
                auto idx_sh = MakeShapeTuple({one, one, index_shape[2]}, span);
                auto idx_raw = emit_load_or_slice(bs, index, idx_ofs, idx_sh, "gather_idx_raw");
                auto idx_row = reshape_to(bs, idx_raw, {one, index_shape[2]}, "gather_idx_row");

                // flat_idx[k] = idx_row[k] * S2 + k  →  selects inp_flat[flat_idx[k]].
                auto idx_sc = emit_to(bs, "tile.muls", {idx_row, make_i32(S2)}, {}, "gather_idx_s");
                auto flat_idx = emit_to(bs, "tile.add", {idx_sc, range_1d}, {}, "gather_fidx");

                return single_row_gather(bs, inp_flat, flat_idx, I2, "gather_row");
              });

          // Reshape [I0*I1, I2] is already the correct 2D layout; prevents Phase 3 optimization.
          auto out_2d = reshape_to(prologue, result, {make_idx(I0I1), make_idx(I2)}, "gather_out");
          return ConversionResult{std::move(prologue), out_2d};
        }
      });

  // tensor.gather_compare → tile.gather_compare
  // Bridges input tensor into a Vec tile, synthesizes the UINT8 tmp workspace
  // tile, and emits a single tuple-typed `tile.gather_compare` call. kvalue is
  // a scalar threshold and passes through unchanged. The dst (gathered indices)
  // and cdst (per-row match counts) tile types are deduced into the call's
  // TupleType output; downstream init_memref allocates the backing buffers
  // from that TupleType.
  std::unordered_map<size_t, InputSpaceReq> gc_input_reqs = {
      {0, {MemorySpace::Vec, std::nullopt}},
  };
  RegisterCustom(
      "tensor.gather_compare",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 2, span)
            << "tensor.gather_compare conversion expects 2 args (input, kvalue), got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();

        auto src_tile = As<TileType>(args[0]->GetType());
        INTERNAL_CHECK_SPAN(src_tile, span)
            << "tensor.gather_compare conversion: input must be Vec tile after bridge, got "
            << args[0]->GetType()->TypeName();
        INTERNAL_CHECK_SPAN(src_tile->shape_.size() == 2, span)
            << "tensor.gather_compare conversion: input must be 2D, got rank " << src_tile->shape_.size();

        std::vector<StmtPtr> prologue;

        // Synthesize tmp: UINT8 workspace shaped like src (1 byte per element).
        // Final size requirement is enforced by the codegen kernel.
        auto tmp_shape_tuple = std::make_shared<MakeTuple>(src_tile->shape_, span);
        std::vector<std::pair<std::string, std::any>> tmp_create_kwargs = {
            {"dtype", DataType(DataType::UINT8)}, {"target_memory", MemorySpace::Vec}};
        auto tmp_create = op_reg.Create("tile.create", {tmp_shape_tuple}, tmp_create_kwargs, span);
        auto tmp_var = std::make_shared<Var>("gc_tmp", tmp_create->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(tmp_var, tmp_create, span));

        // Forward only kwargs the tile op understands (cmp_mode, offset, out_cols, count_dtype).
        std::vector<std::pair<std::string, std::any>> tile_kwargs;
        tile_kwargs.reserve(kwargs.size());
        for (const auto& [k, v] : kwargs) {
          if (k == "cmp_mode" || k == "offset" || k == "out_cols" || k == "count_dtype") {
            tile_kwargs.emplace_back(k, v);
          }
        }

        std::vector<ExprPtr> tile_args = {args[0], args[1], tmp_var};
        auto tile_call = op_reg.Create("tile.gather_compare", tile_args, tile_kwargs, span);
        return ConversionResult{std::move(prologue), tile_call};
      },
      std::move(gc_input_reqs));
}

// ============================================================================
// Paged gather lowering: tensor.paged_gather -> fully-scalar per-row GM->on-chip
// load loop on the Cube core.
//
// For each gathered row i (i in [0, rows), rows = runtime indices count):
//   idx   = indices[i]                         (scalar GM read -> pto.load_scalar)
//   phys  = block_table[idx / block_size] * block_size + idx % block_size  (scalar)
//   row   = tile.load(src, [phys, col_off], [1, size], target_memory=space)  (GM->L1/UB)
//   acc'  = tile.assemble(acc, row, [i, 0])     (write row i into the on-chip tile)
// The bulk KV data goes GM->L1 directly (never UB); only the small index/page-table
// metadata is scalar-read from GM. Mirrors gitcode.com/cann/pypto experimental
// GatherInL1 (OpCoreType::AIC). The accumulator is sized statically by max_indices.
// ============================================================================
void OpConversionRegistry::RegisterPagedGatherOps() {
  RegisterCustom(
      "tensor.paged_gather",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 3, span)
            << "tensor.paged_gather conversion expects 3 args (src, indices, block_table), but got "
            << args.size();
        auto& op_reg = OpRegistry::GetInstance();
        const auto& src = args[0];
        const auto& indices = args[1];
        const auto& block_table = args[2];

        auto src_type = As<TensorType>(src->GetType());
        INTERNAL_CHECK_SPAN(src_type, span) << "tensor.paged_gather: src must be TensorType";
        auto idx_type = As<TensorType>(indices->GetType());
        INTERNAL_CHECK_SPAN(idx_type, span) << "tensor.paged_gather: indices must be TensorType";
        auto bt_type = As<TensorType>(block_table->GetType());
        INTERNAL_CHECK_SPAN(bt_type, span) << "tensor.paged_gather: block_table must be TensorType";

        int block_size = 0;
        int size = 0;
        int col_off = 0;
        int max_indices = 0;
        bool is_trans = false;
        MemorySpace space = MemorySpace::Mat;
        for (const auto& [k, v] : kwargs) {
          if (k == "block_size") {
            block_size = AnyCast<int>(v, "block_size");
          } else if (k == "size") {
            size = AnyCast<int>(v, "size");
          } else if (k == "col_off") {
            col_off = AnyCast<int>(v, "col_off");
          } else if (k == "max_indices") {
            max_indices = AnyCast<int>(v, "max_indices");
          } else if (k == "is_trans") {
            is_trans = AnyCast<bool>(v, "is_trans");
          } else if (k == "space") {
            space = AnyCast<MemorySpace>(v, "space");
          }
        }
        INTERNAL_CHECK_SPAN(block_size > 0 && size > 0 && max_indices > 0, span)
            << "tensor.paged_gather: block_size/size/max_indices must be set and positive";
        INTERNAL_CHECK_SPAN(!is_trans || space == MemorySpace::Mat, span)
            << "tensor.paged_gather: is_trans=true requires space=Mat (L1)";

        const DataType dtype = src_type->dtype_;
        auto make_idx = [&](int64_t value) -> ExprPtr {
          return std::make_shared<ConstInt>(value, DataType::INDEX, span);
        };
        auto make_tuple = [&](std::vector<ExprPtr> elems) -> ExprPtr {
          return std::make_shared<MakeTuple>(std::move(elems), span);
        };
        ExprPtr zero = make_idx(0);
        ExprPtr one = make_idx(1);
        ExprPtr bs = make_idx(block_size);

        std::vector<StmtPtr> prologue;

        // Runtime gathered-row count = last dim of indices (1D: [n]; 2D: [1, n]).
        // Read it via tensor.dim so the loop bound is a proper SSA scalar (works for
        // both static and dynamic index shapes and survives print/parse round-trip).
        const int64_t rows_axis = (idx_type->shape_.size() == 1) ? 0 : 1;
        // Static capacity guard: the accumulator holds at most `max_indices` rows,
        // so a compile-time-known indices length must not exceed it (otherwise
        // tile.gather_row would write out of bounds via dst_offset).
        if (auto rows_const = ir::As<ir::ConstInt>(idx_type->shape_[rows_axis])) {
          CHECK_SPAN(rows_const->value_ <= max_indices, span)
              << "tensor.paged_gather: indices length (" << rows_const->value_ << ") exceeds max_indices ("
              << max_indices << ")";
        }
        auto rows_call = op_reg.Create("tensor.dim", {indices, make_idx(rows_axis)}, span);
        auto rows = std::make_shared<Var>("pg_rows", rows_call->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(rows, rows_call, span));

        // 1) Allocate the static on-chip accumulator tile.
        std::vector<ExprPtr> acc_shape = is_trans
                                             ? std::vector<ExprPtr>{make_idx(size), make_idx(max_indices)}
                                             : std::vector<ExprPtr>{make_idx(max_indices), make_idx(size)};
        std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", dtype},
                                                                       {"target_memory", space}};
        auto acc_create =
            op_reg.Create("tile.create", {MakeShapeTuple(acc_shape, span)}, create_kwargs, span);
        auto acc_var = std::make_shared<Var>("paged_gather_acc", acc_create->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(acc_var, acc_create, span));

        // 2) Per-row loop carrying the accumulator tile.
        auto loop_var = std::make_shared<Var>("pg_i", std::make_shared<ScalarType>(DataType::INDEX), span);
        auto iter_arg = std::make_shared<IterArg>("paged_gather_iter", acc_create->GetType(), acc_var, span);
        auto return_var = std::make_shared<Var>("paged_gather_result", acc_create->GetType(), span);

        std::vector<StmtPtr> body;
        auto bind = [&](const std::string& name_hint, const ExprPtr& expr) -> ExprPtr {
          auto var = std::make_shared<Var>(name_hint, expr->GetType(), span);
          body.push_back(std::make_shared<AssignStmt>(var, expr, span));
          return var;
        };

        // idx = indices[i] (scalar GM read), cast to INDEX for address math.
        ExprPtr idx_read = (idx_type->shape_.size() == 1)
                               ? op_reg.Create("tensor.read", {indices, make_tuple({loop_var})}, span)
                               : op_reg.Create("tensor.read", {indices, make_tuple({zero, loop_var})}, span);
        auto idx_raw = bind("pg_idx_raw", idx_read);
        auto idx_i = bind("pg_idx", MakeCast(idx_raw, DataType::INDEX, span));
        auto block_id = bind("pg_blk", MakeFloorDiv(idx_i, bs, span));
        auto rem = bind("pg_rem", MakeFloorMod(idx_i, bs, span));

        // pblk = block_table[block_id] (scalar GM read), cast to INDEX.
        ExprPtr pblk_read =
            (bt_type->shape_.size() == 1)
                ? op_reg.Create("tensor.read", {block_table, make_tuple({block_id})}, span)
                : op_reg.Create("tensor.read", {block_table, make_tuple({zero, block_id})}, span);
        auto pblk_raw = bind("pg_pblk_raw", pblk_read);
        auto pblk = bind("pg_pblk", MakeCast(pblk_raw, DataType::INDEX, span));
        auto phys = bind("pg_phys", MakeAdd(MakeMul(pblk, bs, span), rem, span));

        // acc' = tile.gather_row(acc, src, dst_offset, src_offset, [1, size]).
        // Loads the physical GM row straight into the accumulator sub-region via
        // pto.subview + pto.tload (GM -> on-chip, no pto.tmov). DPS: writes acc in
        // place, so the loop-carried chain (init / iter_arg / yield / return_var)
        // keeps the accumulator's TileType throughout. is_trans places the row as a
        // column at [0, i]; otherwise as a row at [i, 0].
        ExprPtr dst_offset = is_trans ? make_tuple({zero, loop_var}) : make_tuple({loop_var, zero});
        ExprPtr src_offset = make_tuple({phys, make_idx(col_off)});
        auto row_shapes = MakeShapeTuple({one, make_idx(size)}, span);
        std::vector<std::pair<std::string, std::any>> gr_kwargs = {{"transpose", is_trans}};
        auto gr_call = op_reg.Create("tile.gather_row", {iter_arg, src, dst_offset, src_offset, row_shapes},
                                     gr_kwargs, span);
        auto gr_var = bind("pg_row", gr_call);
        body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{gr_var}, span));

        auto loop_body = SeqStmts::Flatten(std::move(body), span);
        auto for_stmt =
            std::make_shared<ForStmt>(loop_var, zero, rows, one, std::vector<IterArgPtr>{iter_arg}, loop_body,
                                      std::vector<VarPtr>{return_var}, span);
        prologue.push_back(for_stmt);
        return ConversionResult{std::move(prologue), return_var};
      });

  // tensor.create_l1 -> tile.create(target_memory=Mat). The on-chip accumulator
  // for a kernel-driven paged gather_row; deduced as a TensorType at the tensor
  // level so it composes with tensor.matmul, lowered to an L1 (Mat) tile here.
  RegisterCustom(
      "tensor.create_l1",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 1, span)
            << "tensor.create_l1 conversion expects 1 arg (shape), but got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();
        DataType dtype = DataType::FP32;
        bool transpose = false;
        for (const auto& [k, v] : kwargs) {
          if (k == "dtype") dtype = AnyCast<DataType>(v, "dtype");
          if (k == "transpose") transpose = AnyCast<bool>(v, "transpose");
        }
        std::vector<std::pair<std::string, std::any>> create_kwargs = {
            {"dtype", dtype}, {"target_memory", MemorySpace::Mat}, {"transpose", transpose}};
        auto create_call = op_reg.Create("tile.create", {args[0]}, create_kwargs, span);
        return ConversionResult{create_call};
      });

  // tensor.gather_row -> tile.gather_row (DPS, GM->on-chip per-row tload). The
  // accumulator (args[0]) has already been converted to a Mat tile; src (args[1])
  // stays GM (tensor.gather_row is in kSelfLoadingOps so it is not auto-bridged).
  RegisterCustom(
      "tensor.gather_row",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 5, span)
            << "tensor.gather_row conversion expects 5 args (acc, src, dst_offset, src_offset, shapes), "
               "but got "
            << args.size();
        auto& op_reg = OpRegistry::GetInstance();
        bool transpose = false;
        for (const auto& [k, v] : kwargs) {
          if (k == "transpose") transpose = AnyCast<bool>(v, "transpose");
        }
        std::vector<std::pair<std::string, std::any>> gr_kwargs = {{"transpose", transpose}};
        auto gr_call =
            op_reg.Create("tile.gather_row", {args[0], args[1], args[2], args[3], args[4]}, gr_kwargs, span);
        return ConversionResult{gr_call};
      });
}

// ============================================================================
// Scatter lowering (mirror of gather, no compare-form).
//
// tensor.scatter (index form, MVP — rank-2, dim=-1 column scatter):
//   The framework auto-bridges (input, index, src) to Vec tiles via input_reqs.
//   We build the flattened per-element destination index from the column index
//   (flat_idx = index + i*dst_cols) and emit tile.scatter against the tile-level
//   DPS signature (dst, src, indexes), then reconstruct the DPS preserve with a
//   select blend. The surrounding pass wraps the tile result in a tile.store to
//   the output tensor param.
//
// tensor.scatter_mask: same idea — pto.tscatter (mask form) zero-fills the whole
//   dst before writing the selected columns, so it does not preserve dst either.
//   We reconstruct DPS preserve with the same zeroed-scatter + mask + select
//   blend, which also makes chaining two patterns into one dst sound.
// ============================================================================

void OpConversionRegistry::RegisterScatterOps() {
  std::unordered_map<size_t, InputSpaceReq> scatter_input_reqs = {
      {0, {MemorySpace::Vec, std::nullopt}},
      {1, {MemorySpace::Vec, std::nullopt}},
      {2, {MemorySpace::Vec, std::nullopt}},
  };
  RegisterCustom(
      "tensor.scatter",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 3, span)
            << "tensor.scatter conversion expects 3 args (input, index, src), got " << args.size();

        // Validate dim — MVP supports dim=-1 only (column scatter).
        int dim_val = GetKwargOr<int>(kwargs, "dim", -1);
        auto input_tile = As<TileType>(args[0]->GetType());
        auto idx_tile = As<TileType>(args[1]->GetType());
        auto src_tile = As<TileType>(args[2]->GetType());
        INTERNAL_CHECK_SPAN(input_tile && idx_tile && src_tile, span)
            << "tensor.scatter conversion: input/index/src must be Vec tiles after bridge";
        const int rank = static_cast<int>(input_tile->shape_.size());
        const int norm_dim = dim_val < 0 ? dim_val + rank : dim_val;
        INTERNAL_CHECK_SPAN(rank == 2 && norm_dim == rank - 1, span)
            << "tensor.scatter conversion currently supports rank-2 input with dim=-1 only, got "
            << "rank=" << rank << " dim=" << dim_val;

        // The hardware pto.tscatter scatters per *element* using flattened
        // destination indices: dst.flat[idx[i,j]] = src[i,j], looping over the
        // `idx` tile's valid [rows, cols]. The tensor-level API exposes the
        // gather-style column index (same shape as src): index[i,j] is the
        // destination *column* of src[i,j], with the row preserved (the
        // column-wise inverse of gather, out[i, index[i,j]] = src[i,j]). We
        // therefore turn the column index into the flattened index the hardware
        // expects by adding each row's flat base offset:
        //
        //   flat_idx[i, j] = i * dst_cols + index[i, j]
        //
        // Built via row_base[i] = i * dst_cols broadcast across columns:
        //   flat_idx = index + row_base   (row-broadcast add)
        auto& op_reg = OpRegistry::GetInstance();
        auto src_rows = As<ConstInt>(src_tile->shape_[0]);
        // `cols` is the flat-layout column count of the scattered destination.
        // input/output and the inner tile.scatter dst all share this width (S),
        // so reading it off `input` is equivalent to the dst column count used
        // in the `i * dst_cols` flat-index formula above.
        auto dst_cols_c = As<ConstInt>(input_tile->shape_[1]);
        INTERNAL_CHECK_SPAN(src_rows && dst_cols_c, span)
            << "tensor.scatter conversion requires static src rows and dst cols for index expansion";
        const int64_t n = src_rows->value_;
        const int64_t cols = dst_cols_c->value_;
        // Build the flat-index arithmetic in the index tile's own dtype so it
        // keeps satisfying the pto.tscatter element-size matching rule (INT32
        // for 4-byte dst, INT16 for 2/1-byte dst).
        const DataType idx_dtype = idx_tile->dtype_;

        // INT16 flat-index range guard. For 2-byte element dtypes the flattened
        // destination indices are INT16, whose largest representable value is
        // 32767. The biggest index this lowering produces is
        // (n-1)*cols + (cols-1) == n*cols - 1, so n*cols must stay <= 32768.
        // 4-byte dtypes use INT32 indices and are effectively unbounded here.
        // Without this check an oversized tile would silently overflow INT16 and
        // scatter to wrong addresses instead of failing loudly.
        if (idx_dtype == DataType::INT16) {
          // Bound via division so the product never overflows int64_t: rows is
          // capped so rows*cols stays <= 32768. cols is always > 0 here (a
          // 2-byte tile has at least one column), but guard against 0 anyway.
          const int64_t kMaxFlat = 32768;
          const int64_t max_rows = cols == 0 ? kMaxFlat : kMaxFlat / cols;
          CHECK_SPAN(n <= max_rows, span)
              << "tensor.scatter with element dtype " << input_tile->dtype_.ToString()
              << " uses INT16 flattened indices, but the destination is too large: rows(" << n << ") * cols("
              << cols << ") exceeds the INT16 index range (max flat index 32767, rows <= " << max_rows
              << "). Use a smaller tile or split the scatter into chunks.";
        }

        auto make_idx = [&](int64_t v) -> ExprPtr {
          return std::make_shared<ConstInt>(v, DataType::INDEX, span);
        };
        auto make_idx_dt = [&](int64_t v) -> ExprPtr {
          return std::make_shared<ConstInt>(v, idx_dtype, span);
        };
        ExprPtr one = make_idx(1);
        // tile.ci carries both attrs (dtype, descending); supply both so the
        // emitted Call matches the parser-reconstructed one on round-trip.
        std::vector<std::pair<std::string, std::any>> ci_kw = {{"dtype", idx_dtype}, {"descending", false}};

        std::vector<StmtPtr> prologue;
        auto emit = [&](const std::string& op_name, const std::vector<ExprPtr>& op_args,
                        const std::vector<std::pair<std::string, std::any>>& op_kwargs,
                        const std::string& name) -> VarPtr {
          auto call = op_kwargs.empty() ? op_reg.Create(op_name, op_args, span)
                                        : op_reg.Create(op_name, op_args, op_kwargs, span);
          auto var = std::make_shared<Var>(name, call->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(var, call, span));
          return var;
        };
        auto reshape_to = [&](const VarPtr& src, const std::vector<ExprPtr>& shape,
                              const std::string& name) -> VarPtr {
          return emit("tile.reshape", {src, MakeShapeTuple(shape, span)}, {}, name);
        };

        // flat_idx[i, j] = i * dst_cols + index[i, j], built from a per-row base
        // offset broadcast across the columns.
        ExprPtr flat_idx;
        if (n == 1) {
          // Single source row: the only row base offset is 0 * dst_cols = 0, so
          // the flat index equals the column index directly. Emitting the row
          // arange here would create a tile.ci of shape [1, 1], which trips the
          // pto.tci "innermost dim (Cols) != 1" ISA constraint (see
          // DeduceTileCiType in src/ir/op/tile_ops/memory.cpp). That constraint
          // is about column vectors, not a 1-row scatter, so skip the arange.
          flat_idx = args[1];
        } else {
          // row_arange[i] = i  (contiguous arange reshaped to [N, 1]).
          auto row_flat = emit("tile.ci", {make_idx_dt(0), MakeShapeTuple({one, make_idx(n)}, span)}, ci_kw,
                               "scatter_ci_rows");
          auto row_ar = reshape_to(row_flat, {make_idx(n), one}, "scatter_row_arange");
          // row_base[i] = i * dst_cols  (shape [N, 1]).
          auto row_base = emit("tile.muls", {row_ar, make_idx_dt(cols)}, {}, "scatter_row_base");
          // flat_idx[i, j] = index[i, j] + row_base[i]  (row-broadcast add → [N, K]).
          flat_idx = emit("tile.row_expand_add", {args[1], row_base}, {}, "scatter_flat_idx");
        }

        // pto.tscatter only writes the scattered positions and does NOT preserve
        // the destination's other elements (its `dst` operand is treated as
        // write-only), so unwritten coordinates would read back as garbage/zero
        // instead of keeping `input`. PTOAS confirmed this is the machine
        // instruction's behaviour (dst is not RMW), so we permanently
        // reconstruct the preserve semantics on the PyPTO side with a select —
        // which, unlike a multiply-based blend (`input * mask`), emits no
        // pto.tmul (A2/A3 tmul rejects bf16/i8):
        //
        //   scattered = scatter(zeros,   src,  flat_idx)  # src @written, 0 @unwritten
        //   mask      = scatter(zeros_m, ones,  flat_idx)  # 1   @written, 0 @unwritten
        //   pred      = (mask != 0)                         # packed predicate
        //   out       = sel(pred, scattered, input)         # written→scattered, else→input
        //
        // Both scatters MUST use a zeroed base: pto.tscatter zeros unwritten
        // slots, so only a zero base survives to mark the unwritten positions.
        const DataType dt = input_tile->dtype_;
        auto dst_rows_c = As<ConstInt>(input_tile->shape_[0]);
        auto src_cols_c = As<ConstInt>(src_tile->shape_[1]);
        INTERNAL_CHECK_SPAN(dst_rows_c && src_cols_c, span)
            << "tensor.scatter conversion requires static dst rows and src cols for the preserve blend";
        const int64_t m = dst_rows_c->value_;
        const int64_t k = src_cols_c->value_;

        // The mask scatter reuses `flat_idx`, so its element size must match
        // `dt` (else the pto.tscatter index-size rule breaks); within that size
        // pick a compare-friendly type so tile.cmps is well-defined for any
        // supported `dt` (e.g. bf16 → f16 mask).
        const int dt_bytes = static_cast<int>(dt.GetBit()) / 8;
        const DataType mask_dt = (dt_bytes == 4)   ? DataType(DataType::FP32)
                                 : (dt_bytes == 2) ? DataType(DataType::FP16)
                                                   : DataType(DataType::INT8);

        auto make_full = [&](const DataType& fdt, int64_t rows, int64_t cols_, double v,
                             const std::string& name) -> VarPtr {
          ExprPtr val = fdt.IsFloat()
                            ? ExprPtr(std::make_shared<ConstFloat>(v, fdt, span))
                            : ExprPtr(std::make_shared<ConstInt>(static_cast<int64_t>(v), fdt, span));
          std::vector<std::pair<std::string, std::any>> full_kw = {{"dtype", fdt}};
          return emit("tile.full", {MakeShapeTuple({make_idx(rows), make_idx(cols_)}, span), val}, full_kw,
                      name);
        };

        // scattered[*, *] = scatter src into a zeroed [dst_rows, dst_cols] base.
        auto values_zero = make_full(dt, m, cols, 0.0, "scatter_values_zero");
        auto scattered = emit("tile.scatter", {values_zero, args[2], flat_idx}, {}, "scatter_values");
        // mask[*, *] = scatter ones into a zeroed base (1 @written, 0 @unwritten).
        auto mask_zero = make_full(mask_dt, m, cols, 0.0, "scatter_mask_zero");
        auto ones_src = make_full(mask_dt, n, k, 1.0, "scatter_ones");
        auto mask = emit("tile.scatter", {mask_zero, ones_src, flat_idx}, {}, "scatter_mask");
        // pred = (mask != 0)  → packed predicate mask (NE = cmp_type 1).
        ExprPtr zero_scalar = mask_dt.IsFloat() ? ExprPtr(std::make_shared<ConstFloat>(0.0, mask_dt, span))
                                                : ExprPtr(std::make_shared<ConstInt>(0, mask_dt, span));
        std::vector<std::pair<std::string, std::any>> cmp_kw = {{"cmp_type", 1}};
        auto pred = emit("tile.cmps", {mask, zero_scalar}, cmp_kw, "scatter_pred");
        // tmp = TSEL scratch tile (UINT8 [1, 32]).
        std::vector<std::pair<std::string, std::any>> tmp_kw = {{"dtype", DataType(DataType::UINT8)},
                                                                {"target_memory", MemorySpace::Vec}};
        auto tmp =
            emit("tile.create", {MakeShapeTuple({one, make_idx(32)}, span)}, tmp_kw, "scatter_sel_tmp");
        // out = sel(pred, scattered, input, tmp): scattered @written, input @unwritten.
        auto out_call = op_reg.Create("tile.sel", {pred, scattered, args[0], tmp}, span);
        return ConversionResult{std::move(prologue), out_call};
      },
      std::move(scatter_input_reqs));

  std::unordered_map<size_t, InputSpaceReq> scatter_mask_input_reqs = {
      {0, {MemorySpace::Vec, std::nullopt}},
      {1, {MemorySpace::Vec, std::nullopt}},
  };
  RegisterCustom(
      "tensor.scatter_mask",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 2, span)
            << "tensor.scatter_mask conversion expects 2 args (input, dst), got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();
        auto input_tile = As<TileType>(args[0]->GetType());
        auto dst_tile = As<TileType>(args[1]->GetType());
        INTERNAL_CHECK_SPAN(input_tile && dst_tile, span)
            << "tensor.scatter_mask conversion: input/dst must be Vec tiles after bridge";

        // pto.tscatter (mask form) zero-fills the entire dst tile before writing
        // the mask-selected columns (TScatterMaskImpl calls InitUBBuffer), so it
        // does NOT preserve dst's unselected columns — they read back as 0. To
        // honour the DPS preserve contract (and make chaining two patterns into
        // one dst sound), reconstruct preserve on the PyPTO side with the same
        // zeroed-scatter + mask + select blend as the index form:
        //
        //   scattered = scatter_mask(zeros,   input)  # input @selected, 0 @unselected
        //   mask      = scatter_mask(zeros_m, ones)   # 1     @selected, 0 @unselected
        //   pred      = (mask != 0)                    # packed predicate
        //   out       = sel(pred, scattered, dst)      # selected→scattered, else→dst
        auto in_rows = As<ConstInt>(input_tile->shape_[0]);
        auto in_cols = As<ConstInt>(input_tile->shape_[1]);
        auto dst_cols_c = As<ConstInt>(dst_tile->shape_[1]);
        INTERNAL_CHECK_SPAN(in_rows && in_cols && dst_cols_c, span)
            << "tensor.scatter_mask conversion requires static shapes for the preserve blend";
        const int64_t b = in_rows->value_;
        const int64_t c = in_cols->value_;
        const int64_t dst_cols = dst_cols_c->value_;
        const DataType dt = dst_tile->dtype_;

        // Mask blend dtype: a compare-friendly type within dst's element size
        // (bf16 → f16) so tile.cmps is well-defined for any supported dtype.
        const int dt_bytes = static_cast<int>(dt.GetBit()) / 8;
        const DataType mask_dt = (dt_bytes == 4)   ? DataType(DataType::FP32)
                                 : (dt_bytes == 2) ? DataType(DataType::FP16)
                                                   : DataType(DataType::INT8);

        auto make_idx = [&](int64_t v) -> ExprPtr {
          return std::make_shared<ConstInt>(v, DataType::INDEX, span);
        };
        std::vector<StmtPtr> prologue;
        auto emit = [&](const std::string& op_name, const std::vector<ExprPtr>& op_args,
                        const std::vector<std::pair<std::string, std::any>>& op_kwargs,
                        const std::string& name) -> VarPtr {
          auto call = op_kwargs.empty() ? op_reg.Create(op_name, op_args, span)
                                        : op_reg.Create(op_name, op_args, op_kwargs, span);
          auto var = std::make_shared<Var>(name, call->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(var, call, span));
          return var;
        };
        auto make_full = [&](const DataType& fdt, int64_t rows, int64_t cols_, double v,
                             const std::string& name) -> VarPtr {
          ExprPtr val = fdt.IsFloat()
                            ? ExprPtr(std::make_shared<ConstFloat>(v, fdt, span))
                            : ExprPtr(std::make_shared<ConstInt>(static_cast<int64_t>(v), fdt, span));
          return emit("tile.full", {MakeShapeTuple({make_idx(rows), make_idx(cols_)}, span), val},
                      {{"dtype", fdt}}, name);
        };

        // scattered = input written into the mask-selected columns of a zeroed dst.
        auto values_zero = make_full(dt, b, dst_cols, 0.0, "scatter_mask_values_zero");
        auto scattered = emit("tile.scatter_mask", {values_zero, args[0]}, kwargs, "scatter_mask_values");
        // mask = ones written into the same selected columns of a zeroed base.
        auto mask_zero = make_full(mask_dt, b, dst_cols, 0.0, "scatter_mask_mask_zero");
        auto ones_src = make_full(mask_dt, b, c, 1.0, "scatter_mask_ones");
        auto mask = emit("tile.scatter_mask", {mask_zero, ones_src}, kwargs, "scatter_mask_mask");
        // pred = (mask != 0)  → packed predicate (NE = cmp_type 1).
        ExprPtr zero_scalar = mask_dt.IsFloat() ? ExprPtr(std::make_shared<ConstFloat>(0.0, mask_dt, span))
                                                : ExprPtr(std::make_shared<ConstInt>(0, mask_dt, span));
        auto pred = emit("tile.cmps", {mask, zero_scalar}, {{"cmp_type", 1}}, "scatter_mask_pred");
        // tmp = TSEL scratch tile (UINT8 [1, 32]).
        auto tmp = emit("tile.create", {MakeShapeTuple({make_idx(1), make_idx(32)}, span)},
                        {{"dtype", DataType(DataType::UINT8)}, {"target_memory", MemorySpace::Vec}},
                        "scatter_mask_sel_tmp");
        // out = sel(pred, scattered, dst, tmp): scattered @selected, dst @unselected.
        auto out_call = op_reg.Create("tile.sel", {pred, scattered, args[1], tmp}, span);
        return ConversionResult{std::move(prologue), out_call};
      },
      std::move(scatter_mask_input_reqs));
}

// ============================================================================
// Tensor compare op: lower to packed mask + tile.full(one/zero) + tile.sel.
// Dispatches to tile.cmp (tensor-vs-tensor) or tile.cmps (tensor-vs-scalar)
// based on the rhs operand type — there is no tensor.cmps front-end op.
// ============================================================================

void OpConversionRegistry::RegisterCmpOps() {
  auto CmpConv = [](const std::vector<ExprPtr>& args,
                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                    const Span& span) -> ConversionResult {
    auto& op_reg = OpRegistry::GetInstance();
    auto lhs_tile = As<TileType>(args[0]->GetType());
    INTERNAL_CHECK_SPAN(lhs_tile, span)
        << "tensor.cmp conversion: lhs must be TileType after memory promotion, got "
        << args[0]->GetType()->TypeName();

    std::string tile_cmp_op;
    std::vector<ExprPtr> result_shape;
    DataType result_dtype = lhs_tile->dtype_;
    if (auto rhs_tile = As<TileType>(args[1]->GetType())) {
      tile_cmp_op = "tile.cmp";
      auto broadcast_result = BroadcastShapes(lhs_tile->shape_, rhs_tile->shape_);
      INTERNAL_CHECK_SPAN(broadcast_result.success, span)
          << "tensor.cmp conversion: incompatible shapes " << FormatShape(lhs_tile->shape_) << " and "
          << FormatShape(rhs_tile->shape_);
      result_shape = broadcast_result.shape;
      auto promoted = PromoteDataTypes(lhs_tile->dtype_, rhs_tile->dtype_);
      INTERNAL_CHECK_SPAN(promoted, span)
          << "tensor.cmp conversion: incompatible dtypes " << lhs_tile->dtype_.ToString() << " and "
          << rhs_tile->dtype_.ToString();
      result_dtype = *promoted;
    } else if (auto rhs_scalar = As<ScalarType>(args[1]->GetType())) {
      tile_cmp_op = "tile.cmps";
      result_shape = lhs_tile->shape_;
      auto promoted = PromoteDataTypes(lhs_tile->dtype_, rhs_scalar->dtype_);
      INTERNAL_CHECK_SPAN(promoted, span)
          << "tensor.cmp conversion: incompatible dtypes " << lhs_tile->dtype_.ToString() << " and "
          << rhs_scalar->dtype_.ToString();
      result_dtype = *promoted;
    } else {
      INTERNAL_UNREACHABLE_SPAN(span) << "tensor.cmp conversion: rhs must be TileType or ScalarType, got "
                                      << args[1]->GetType()->TypeName();
    }

    std::vector<StmtPtr> prologue;

    auto mask_call = op_reg.Create(tile_cmp_op, args, kwargs, span);
    auto mask_var = std::make_shared<Var>("cmp_mask", mask_call->GetType(), span);
    prologue.push_back(std::make_shared<AssignStmt>(mask_var, mask_call, span));

    auto shape_tuple = std::make_shared<MakeTuple>(result_shape, span);
    auto make_full = [&](double v, const std::string& name) {
      std::vector<std::pair<std::string, std::any>> kw = {{"dtype", result_dtype}};
      ExprPtr val = result_dtype.IsFloat()
                        ? ExprPtr(std::make_shared<ConstFloat>(v, result_dtype, span))
                        : ExprPtr(std::make_shared<ConstInt>(static_cast<int64_t>(v), result_dtype, span));
      auto call = op_reg.Create("tile.full", {shape_tuple, val}, kw, span);
      auto var = std::make_shared<Var>(name, call->GetType(), span);
      prologue.push_back(std::make_shared<AssignStmt>(var, call, span));
      return var;
    };
    auto one_var = make_full(1.0, "cmp_one");
    auto zero_var = make_full(0.0, "cmp_zero");

    std::vector<ExprPtr> tmp_shape_dims = {std::make_shared<ConstInt>(1, DataType::INDEX, span),
                                           std::make_shared<ConstInt>(32, DataType::INDEX, span)};
    auto tmp_shape_tuple = std::make_shared<MakeTuple>(tmp_shape_dims, span);
    std::vector<std::pair<std::string, std::any>> tmp_kw = {{"dtype", DataType::UINT8},
                                                            {"target_memory", MemorySpace::Vec}};
    auto tmp_call = op_reg.Create("tile.create", {tmp_shape_tuple}, tmp_kw, span);
    auto tmp_var = std::make_shared<Var>("cmp_tmp", tmp_call->GetType(), span);
    prologue.push_back(std::make_shared<AssignStmt>(tmp_var, tmp_call, span));

    auto sel_call = op_reg.Create("tile.sel", {mask_var, one_var, zero_var, tmp_var}, span);
    return ConversionResult{std::move(prologue), sel_call};
  };

  RegisterCustom("tensor.cmp", CmpConv);
}

// ============================================================================
// Distributed (pld.*) ops: synthesise auxiliary tile scratch buffers so the
// memory allocator assigns UB addresses before codegen (--pto-level=level3)
// ============================================================================

void OpConversionRegistry::RegisterDistributedOps() {
  // pld.tensor.put -> tile.create(stage) + pld.tile.put(dst, peer, src, stage).
  // Stage shape is [rows, cols] with rows = product(leading dims), cols =
  // innermost dim: the 2-D-flattened transfer extent codegen previously
  // synthesized inline. For subregion put, use the explicit transfer shape
  // rather than the full dst window shape and forward offsets/shape to
  // pld.tile.put.
  RegisterCustom(
      "pld.tensor.put",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 3 || args.size() == 6, span)
            << "pld.tensor.put conversion expects 3 args (dst, peer, src) or 6 "
               "(dst, peer, src, dst_offsets, src_offsets, shape), got "
            << args.size();
        auto& op_reg = OpRegistry::GetInstance();

        auto dst_type = As<DistributedTensorType>(args[0]->GetType());
        INTERNAL_CHECK_SPAN(dst_type, span)
            << "pld.tensor.put conversion: dst must be DistributedTensorType, got "
            << args[0]->GetType()->TypeName();
        std::vector<ExprPtr> transfer_shape = dst_type->shape_;
        if (args.size() == 6) {
          auto shape_tuple_arg = As<MakeTuple>(args[5]);
          INTERNAL_CHECK_SPAN(shape_tuple_arg, span) << "pld.tensor.put conversion: shape must be MakeTuple";
          transfer_shape = shape_tuple_arg->elements_;
        }
        INTERNAL_CHECK_SPAN(!transfer_shape.empty(), span)
            << "pld.tensor.put conversion: transfer shape requires rank >= 1";

        // Flatten N-D to [rows, cols]: rows = ∏ leading dims, cols = innermost.
        int64_t cols_val = 0;
        {
          auto last = As<ConstInt>(transfer_shape.back());
          INTERNAL_CHECK_SPAN(last, span)
              << "pld.tensor.put conversion: transfer innermost dimension must be ConstInt";
          cols_val = last->value_;
        }
        int64_t rows_val = 1;
        for (size_t i = 0; i + 1 < transfer_shape.size(); ++i) {
          auto d = As<ConstInt>(transfer_shape[i]);
          INTERNAL_CHECK_SPAN(d, span)
              << "pld.tensor.put conversion: transfer dimension " << i << " must be ConstInt";
          rows_val *= d->value_;
        }
        auto rows_expr = std::make_shared<ConstInt>(rows_val, DataType::INDEX, span);
        auto cols_expr = std::make_shared<ConstInt>(cols_val, DataType::INDEX, span);
        auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{rows_expr, cols_expr}, span);

        std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", dst_type->dtype_},
                                                                       {"target_memory", MemorySpace::Vec}};
        auto create_call = op_reg.Create("tile.create", {shape_tuple}, create_kwargs, span);
        auto stage_var = std::make_shared<Var>("tput_stage", create_call->GetType(), span);
        std::vector<StmtPtr> prologue;
        prologue.push_back(std::make_shared<AssignStmt>(stage_var, create_call, span));

        std::vector<ExprPtr> put_args{args[0], args[1], args[2], stage_var};
        if (args.size() == 6) {
          put_args.insert(put_args.end(), args.begin() + 3, args.end());
        }
        auto put_call = op_reg.Create("pld.tile.put", put_args, kwargs, span);
        return ConversionResult{std::move(prologue), put_call};
      });

  // pld.tensor.get -> tile.create(stage) + pld.tile.get(dst, peer, src, stage).
  // Mirror pld.tensor.put so the TGET VEC bounce buffer is allocated by the IR
  // memory pipeline instead of being synthesized as an unaddressed codegen-only
  // tile. Subregion get uses the explicit transfer shape.
  RegisterCustom(
      "pld.tensor.get",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        INTERNAL_CHECK_SPAN(args.size() == 3 || args.size() == 6, span)
            << "pld.tensor.get conversion expects 3 args (dst, peer, src) or 6 "
               "(dst, peer, src, dst_offsets, src_offsets, shape), got "
            << args.size();
        INTERNAL_CHECK_SPAN(kwargs.empty(), span) << "pld.tensor.get conversion expects no kwargs";
        auto& op_reg = OpRegistry::GetInstance();

        auto dst_type = As<DistributedTensorType>(args[0]->GetType());
        INTERNAL_CHECK_SPAN(dst_type, span)
            << "pld.tensor.get conversion: dst must be DistributedTensorType, got "
            << args[0]->GetType()->TypeName();
        std::vector<ExprPtr> transfer_shape = dst_type->shape_;
        if (args.size() == 6) {
          auto shape_tuple_arg = As<MakeTuple>(args[5]);
          INTERNAL_CHECK_SPAN(shape_tuple_arg, span) << "pld.tensor.get conversion: shape must be MakeTuple";
          transfer_shape = shape_tuple_arg->elements_;
        }
        INTERNAL_CHECK_SPAN(!transfer_shape.empty(), span)
            << "pld.tensor.get conversion: transfer shape requires rank >= 1";

        int64_t cols_val = 0;
        {
          auto last = As<ConstInt>(transfer_shape.back());
          INTERNAL_CHECK_SPAN(last, span)
              << "pld.tensor.get conversion: transfer innermost dimension must be ConstInt";
          cols_val = last->value_;
        }
        int64_t rows_val = 1;
        for (size_t i = 0; i + 1 < transfer_shape.size(); ++i) {
          auto d = As<ConstInt>(transfer_shape[i]);
          INTERNAL_CHECK_SPAN(d, span)
              << "pld.tensor.get conversion: transfer dimension " << i << " must be ConstInt";
          rows_val *= d->value_;
        }
        auto rows_expr = std::make_shared<ConstInt>(rows_val, DataType::INDEX, span);
        auto cols_expr = std::make_shared<ConstInt>(cols_val, DataType::INDEX, span);
        auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{rows_expr, cols_expr}, span);

        std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", dst_type->dtype_},
                                                                       {"target_memory", MemorySpace::Vec}};
        auto create_call = op_reg.Create("tile.create", {shape_tuple}, create_kwargs, span);
        auto stage_var = std::make_shared<Var>("tget_stage", create_call->GetType(), span);
        std::vector<StmtPtr> prologue;
        prologue.push_back(std::make_shared<AssignStmt>(stage_var, create_call, span));

        std::vector<ExprPtr> get_args{args[0], args[1], args[2], stage_var};
        if (args.size() == 6) {
          get_args.insert(get_args.end(), args.begin() + 3, args.end());
        }
        auto get_call = op_reg.Create("pld.tile.get", get_args, span);
        return ConversionResult{std::move(prologue), get_call};
      });
}

void OpConversionRegistry::RegisterSimple(const std::string& from_op, const std::string& to_op,
                                          std::unordered_map<size_t, InputSpaceReq> input_reqs) {
  // Capture to_op by value for the lambda
  ConversionFunc func = [to_op](const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const Span& span) -> ConversionResult {
    auto& reg = OpRegistry::GetInstance();
    CallPtr call;
    if (kwargs.empty()) {
      call = reg.Create(to_op, args, span);
    } else {
      call = reg.Create(to_op, args, kwargs, span);
    }
    return ConversionResult{call};
  };
  conversions_[from_op] = ConversionEntry{std::move(func), std::move(input_reqs)};
}

void OpConversionRegistry::RegisterCustom(const std::string& from_op, ConversionFunc func,
                                          std::unordered_map<size_t, InputSpaceReq> input_reqs) {
  conversions_[from_op] = ConversionEntry{std::move(func), std::move(input_reqs)};
}

const ConversionEntry* OpConversionRegistry::Lookup(const std::string& op_name) const {
  auto it = conversions_.find(op_name);
  if (it == conversions_.end()) {
    return nullptr;
  }
  return &it->second;
}

bool OpConversionRegistry::HasConversion(const std::string& op_name) const {
  return conversions_.count(op_name) > 0;
}

}  // namespace ir
}  // namespace pypto
