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
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

ExprPtr MakeZeroOffsetsTuple(size_t ndim, const Span& span) {
  std::vector<ExprPtr> zeros;
  zeros.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    zeros.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
  }
  return std::make_shared<MakeTuple>(zeros, span);
}

ExprPtr MakeShapesTuple(const std::vector<ExprPtr>& shape, const Span& span) {
  return std::make_shared<MakeTuple>(shape, span);
}

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

// Load a matmul operand into Mat space if it's a TensorType.
// If already a TileType, returns the operand as-is.
ExprPtr LoadOperandToMat(const ExprPtr& operand, bool transpose, const std::string& var_name,
                         std::vector<StmtPtr>& prologue, const Span& span) {
  auto& op_reg = OpRegistry::GetInstance();
  auto tensor_type = As<TensorType>(operand->GetType());
  if (tensor_type) {
    auto offsets = MakeZeroOffsetsTuple(tensor_type->shape_.size(), span);
    auto shape = tensor_type->shape_;
    auto shapes = MakeShapesTuple(shape, span);
    std::vector<std::pair<std::string, std::any>> kw = {{"target_memory", MemorySpace::Mat},
                                                        {"transpose", transpose}};
    auto load = op_reg.Create("tile.load", {operand, offsets, shapes, shapes}, kw, span);
    auto load_var = std::make_shared<Var>(var_name, load->GetType(), span);
    prologue.push_back(std::make_shared<AssignStmt>(load_var, load, span));
    return load_var;
  }
  auto tile_type = As<TileType>(operand->GetType());
  if (tile_type) {
    return operand;
  }
  INTERNAL_CHECK(false) << "LoadOperandToMat: unexpected type: " << operand->GetType()->TypeName();
  return nullptr;  // unreachable
}

}  // namespace

OpConversionRegistry& OpConversionRegistry::GetInstance() {
  static OpConversionRegistry instance;
  return instance;
}

OpConversionRegistry::OpConversionRegistry() {
  auto& reg = OpRegistry::GetInstance();

  // ────────────────────────────────────────────────────────────────────────
  // Simple 1:1 conversions (tensor op → tile op, same args/kwargs)
  // ────────────────────────────────────────────────────────────────────────

  // Scalar ops
  RegisterSimple("tensor.adds", "tile.adds");
  RegisterSimple("tensor.subs", "tile.subs");
  RegisterSimple("tensor.muls", "tile.muls");
  RegisterSimple("tensor.divs", "tile.divs");

  // Unary ops
  RegisterSimple("tensor.neg", "tile.neg");
  RegisterSimple("tensor.recip", "tile.recip");
  RegisterSimple("tensor.exp", "tile.exp");
  RegisterSimple("tensor.sqrt", "tile.sqrt");
  RegisterSimple("tensor.rsqrt", "tile.rsqrt");
  RegisterSimple("tensor.cast", "tile.cast");

  // Broadcast ops
  RegisterSimple("tensor.row_expand_mul", "tile.row_expand_mul");
  RegisterSimple("tensor.row_expand_div", "tile.row_expand_div");
  RegisterSimple("tensor.col_expand_mul", "tile.col_expand_mul");
  RegisterSimple("tensor.row_expand", "tile.row_expand");
  RegisterSimple("tensor.row_expand_add", "tile.row_expand_add");
  RegisterSimple("tensor.row_expand_sub", "tile.row_expand_sub");
  RegisterSimple("tensor.col_expand", "tile.col_expand");
  RegisterSimple("tensor.col_expand_sub", "tile.col_expand_sub");
  RegisterSimple("tensor.col_expand_div", "tile.col_expand_div");
  RegisterSimple("tensor.expands", "tile.expands");

  // Transform ops
  RegisterSimple("tensor.reshape", "tile.reshape");
  RegisterSimple("tensor.transpose", "tile.transpose");
  RegisterSimple("tensor.concat", "tile.concat");

  // Memory creation ops
  RegisterSimple("tensor.full", "tile.full");

  // ────────────────────────────────────────────────────────────────────────
  // Broadcast-aware elementwise binary ops
  //
  // When both operands have the same shape → tile.{op}
  // When one operand is [M,1] (column vector) → tile.row_expand_{op}
  // ────────────────────────────────────────────────────────────────────────

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
  RegisterCustom("tensor.maximum", MakeBroadcastBinaryConv("tile.maximum", "tile.maximum"));

  // ────────────────────────────────────────────────────────────────────────
  // tensor.slice → tile.load (gm_tensor) or tile.slice (local_tensor)
  //
  // gm_tensor.slice(tensor, shape, offset) → tile.load(tensor, offset, shape, shape, target_memory=Vec)
  // local_tensor.slice(tile, shape, offset) → tile.slice(tile, shape, offset)
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.slice",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3 || args.size() == 4)
            << "tensor.slice conversion expects 3 or 4 args (tensor, shape, offset[, valid_shape])";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];
        const auto& shape = args[1];
        const auto& offset = args[2];

        auto tensor_type = As<TensorType>(input->GetType());
        auto tile_type = As<TileType>(input->GetType());

        if (tensor_type) {
          // gm_tensor: function parameter or prior gm_tensor.slice result → tile.load
          auto valid_shapes = (args.size() == 4) ? args[3] : shape;
          std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                       {"transpose", false}};
          auto load_call =
              op_reg.Create("tile.load", {input, offset, shape, valid_shapes}, load_kwargs, span);
          return ConversionResult{load_call};
        }

        if (tile_type) {
          // local_tensor: created via tensor.create (now tile) → tile.slice
          std::vector<ExprPtr> slice_args = {input, shape, offset};
          if (args.size() == 4) {
            slice_args.push_back(args[3]);
          }
          auto slice_call = op_reg.Create("tile.slice", slice_args, span);
          return ConversionResult{slice_call};
        }

        CHECK(false) << "tensor.slice conversion: unexpected input type: " << input->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.matmul → tile.load(Mat) + tile.move(L0A/L0B) + tile.matmul + tile.store
  //
  // tensor.matmul(lhs, rhs, a_trans=False, b_trans=True, c_matrix_nz=False)
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.matmul",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 2) << "tensor.matmul conversion expects 2 args (lhs, rhs)";

        bool a_trans = GetKwargOr<bool>(kwargs, "a_trans", false);
        bool b_trans = GetKwargOr<bool>(kwargs, "b_trans", false);

        std::vector<StmtPtr> prologue;
        auto lhs_mat = LoadOperandToMat(args[0], a_trans, "lhs_mat", prologue, span);
        auto rhs_mat = LoadOperandToMat(args[1], b_trans, "rhs_mat", prologue, span);

        auto matmul_call = OpRegistry::GetInstance().Create("tile.matmul", {lhs_mat, rhs_mat}, span);
        return ConversionResult{std::move(prologue), matmul_call};
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.matmul_acc → tile.matmul_acc
  //
  // tensor.matmul_acc(acc, lhs, rhs, a_trans=False, b_trans=False)
  // acc is passed through (already TileType from IterArg type propagation).
  // lhs/rhs are loaded into Mat space (same as tensor.matmul).
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.matmul_acc",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.matmul_acc conversion expects 3 args (acc, lhs, rhs)";

        bool a_trans = GetKwargOr<bool>(kwargs, "a_trans", false);
        bool b_trans = GetKwargOr<bool>(kwargs, "b_trans", false);

        std::vector<StmtPtr> prologue;
        auto lhs_mat = LoadOperandToMat(args[1], a_trans, "lhs_mat", prologue, span);
        auto rhs_mat = LoadOperandToMat(args[2], b_trans, "rhs_mat", prologue, span);

        auto matmul_acc_call =
            OpRegistry::GetInstance().Create("tile.matmul_acc", {args[0], lhs_mat, rhs_mat}, span);
        return ConversionResult{std::move(prologue), matmul_acc_call};
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.row_max / tensor.row_sum → tile.row_max / tile.row_sum
  //
  // Tile reductions need a tmp_tile with the same shape as input (allocated
  // in Vec space) as a workspace parameter.
  // ────────────────────────────────────────────────────────────────────────

  auto MakeReductionConv = [](const std::string& tile_op) -> ConversionFunc {
    return [tile_op](const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                     const Span& span) -> ConversionResult {
      CHECK(args.size() == 1) << tile_op << " conversion expects 1 arg (input tile)";
      auto& op_reg = OpRegistry::GetInstance();

      const auto& input = args[0];
      auto tile_type = As<TileType>(input->GetType());
      CHECK(tile_type) << tile_op << " conversion: input must be TileType, got "
                       << input->GetType()->TypeName();

      // Build a padded shape for the tmp tile: keep rows, round up cols to a
      // hardware-friendly size (use 128 as default alignment).  If the original
      // last dim is already ConstInt we can compute at compile time; otherwise
      // fall back to the original shape.
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

  RegisterCustom(
      "tensor.fillpad",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 1) << "tensor.fillpad conversion expects 1 arg (input)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        if (As<TileType>(input->GetType())) {
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tile.fillpad", {input}, span)};
          }
          return ConversionResult{op_reg.Create("tile.fillpad", {input}, kwargs, span)};
        }

        auto tensor_type = As<TensorType>(input->GetType());
        CHECK(tensor_type) << "tensor.fillpad conversion: input must be TensorType or TileType, got "
                           << input->GetType()->TypeName();

        auto offsets = MakeZeroOffsetsTuple(tensor_type->shape_.size(), span);
        auto shapes = MakeShapesTuple(tensor_type->shape_, span);

        std::vector<ExprPtr> valid_shape = tensor_type->shape_;
        if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->valid_shape.empty()) {
          valid_shape = tensor_type->tensor_view_->valid_shape;
        }
        auto valid_shapes = MakeShapesTuple(valid_shape, span);

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

  // ────────────────────────────────────────────────────────────────────────
  // tensor.assemble → tile.store
  //
  // tensor.assemble(target, source, offset) → tile.store(source_tile, offset, target)
  // Only converts when source is a TileType. Falls back to pass-through otherwise.
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.assemble",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.assemble conversion expects 3 args (target, source, offset)";
        auto& op_reg = OpRegistry::GetInstance();

        const auto& target = args[0];
        const auto& source = args[1];
        const auto& offset = args[2];

        auto source_tile_type = As<TileType>(source->GetType());
        auto target_tensor_type = As<TensorType>(target->GetType());
        auto target_tile_type = As<TileType>(target->GetType());

        if (source_tile_type && target_tensor_type) {
          // Tile → Tensor: use tile.store
          auto store_call = op_reg.Create("tile.store", {source, offset, target}, span);
          return ConversionResult{store_call};
        }

        if (source_tile_type && target_tile_type) {
          // Both are tiles → tile.assemble(target, source, offset)
          auto assemble_call = op_reg.Create("tile.assemble", {target, source, offset}, span);
          return ConversionResult{assemble_call};
        }

        if (target_tile_type && !source_tile_type) {
          // Target is tile, source is still tensor → load source to Vec, then tile.assemble
          auto source_tensor_type = As<TensorType>(source->GetType());
          CHECK(source_tensor_type) << "tensor.assemble: source must be TensorType or TileType, but got "
                                    << source->GetType()->TypeName();
          std::vector<StmtPtr> prologue;
          auto offsets_load = MakeZeroOffsetsTuple(source_tensor_type->shape_.size(), span);
          auto shapes = MakeShapesTuple(source_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", MemorySpace::Vec},
                                                                   {"transpose", false}};
          auto load_call = op_reg.Create("tile.load", {source, offsets_load, shapes, shapes}, load_kw, span);
          auto source_tile_var = std::make_shared<Var>("assemble_src", load_call->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(source_tile_var, load_call, span));

          auto assemble_call = op_reg.Create("tile.assemble", {target, source_tile_var, offset}, span);
          return ConversionResult{std::move(prologue), assemble_call};
        }

        // Both still tensors — keep as tensor.assemble
        if (kwargs.empty()) {
          return ConversionResult{op_reg.Create("tensor.assemble", args, span)};
        }
        return ConversionResult{op_reg.Create("tensor.assemble", args, kwargs, span)};
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.scatter_update → tile.scatter_update
  //
  // When input is a TileType (local buffer created via tensor.create), load
  // index and src if needed, then emit tile.scatter_update(input, index, src).
  // When input is a TensorType (global memory, e.g. KV cache pool), keep the
  // op unchanged — the orchestration codegen handles it via memcpy.
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.scatter_update",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.scatter_update conversion expects 3 args (input, index, src)";
        auto& op_reg = OpRegistry::GetInstance();

        const auto& input = args[0];
        const auto& index = args[1];
        const auto& src = args[2];

        auto input_tensor_type = As<TensorType>(input->GetType());

        if (input_tensor_type) {
          // Global tensor input — keep as tensor.scatter_update (handled by orchestration codegen)
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tensor.scatter_update", args, span)};
          }
          return ConversionResult{op_reg.Create("tensor.scatter_update", args, kwargs, span)};
        }

        CHECK(As<TileType>(input->GetType()))
            << "tensor.scatter_update: unexpected input type: " << input->GetType()->TypeName();

        std::vector<StmtPtr> prologue;

        // Load index to Vec tile if it is still a global tensor
        ExprPtr index_tile = index;
        if (auto index_tensor_type = As<TensorType>(index->GetType())) {
          auto offsets = MakeZeroOffsetsTuple(index_tensor_type->shape_.size(), span);
          auto shapes = MakeShapesTuple(index_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", MemorySpace::Vec},
                                                                   {"transpose", false}};
          auto load = op_reg.Create("tile.load", {index, offsets, shapes, shapes}, load_kw, span);
          auto idx_var = std::make_shared<Var>("scatter_idx", load->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(idx_var, load, span));
          index_tile = idx_var;
        }

        // Load src to Vec tile if it is still a global tensor
        ExprPtr src_tile = src;
        if (auto src_tensor_type = As<TensorType>(src->GetType())) {
          auto offsets = MakeZeroOffsetsTuple(src_tensor_type->shape_.size(), span);
          auto shapes = MakeShapesTuple(src_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", MemorySpace::Vec},
                                                                   {"transpose", false}};
          auto load = op_reg.Create("tile.load", {src, offsets, shapes, shapes}, load_kw, span);
          auto src_var = std::make_shared<Var>("scatter_src", load->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(src_var, load, span));
          src_tile = src_var;
        }

        auto scatter_call = op_reg.Create("tile.scatter_update", {input, index_tile, src_tile}, kwargs, span);
        return ConversionResult{std::move(prologue), scatter_call};
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.scatter_ → nested ForStmt with tile.read + tile.write
  //
  // For every element position (i0, ..., in) in the index tensor:
  //   idx = tile.read(index, [i0, ..., in])
  //   src_val = tile.read(src, [i0, ..., in])  (or scalar src directly)
  //   Build write_indices: same as (i0, ..., in) but replace dim-th with idx
  //   tile.write(input, write_indices, src_val)
  //
  // For reduce="add":   old = tile.read(input, write_indices); write old + src_val
  // For reduce="multiply": old = tile.read(input, write_indices); write old * src_val
  //
  // Uses nested ForStmt loops (scf.for) which PTOAS compiles to C++ for loops.
  // PTOAS eliminates single-trip scf.for loops, so real multi-iteration loops
  // are required for tgetval/tsetval to compile.
  //
  // When input is a TensorType (global memory), keep the op unchanged for
  // orchestration codegen.
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.scatter_",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.scatter_ conversion expects 3 args (input, index, src)";
        auto& op_reg = OpRegistry::GetInstance();

        const auto& input = args[0];
        const auto& index = args[1];
        const auto& src = args[2];

        // Global tensor input → keep as tensor.scatter_ (orchestration codegen handles it)
        if (As<TensorType>(input->GetType())) {
          return ConversionResult{op_reg.Create("tensor.scatter_", args, kwargs, span)};
        }

        auto input_tile_type = As<TileType>(input->GetType());
        CHECK(input_tile_type) << "tensor.scatter_: unexpected input type: " << input->GetType()->TypeName();

        const size_t rank = input_tile_type->shape_.size();

        // Extract kwargs
        int dim = GetKwargOr<int>(kwargs, "dim", 0);
        std::string reduce = GetKwargOr<std::string>(kwargs, "reduce", std::string("none"));
        // Normalize negative dim
        if (dim < 0) dim += static_cast<int>(rank);
        CHECK(dim >= 0 && dim < static_cast<int>(rank))
            << "tensor.scatter_: dim " << dim << " is out of range for rank " << rank;
        CHECK(reduce == "none" || reduce == "add" || reduce == "multiply")
            << "tensor.scatter_: unsupported reduce mode '" << reduce << "'";

        std::vector<StmtPtr> prologue;

        // Load index to Vec tile if it is still a global tensor
        ExprPtr index_tile = index;
        auto index_tile_type = As<TileType>(index->GetType());
        if (auto index_tensor_type = As<TensorType>(index->GetType())) {
          auto offsets = MakeZeroOffsetsTuple(index_tensor_type->shape_.size(), span);
          auto shapes = MakeShapesTuple(index_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", MemorySpace::Vec},
                                                                   {"transpose", false}};
          auto load = op_reg.Create("tile.load", {index, offsets, shapes, shapes}, load_kw, span);
          auto idx_var = std::make_shared<Var>("scatter_idx", load->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(idx_var, load, span));
          index_tile = idx_var;
          index_tile_type = As<TileType>(load->GetType());
        }

        // Determine whether src is scalar
        bool src_is_scalar = static_cast<bool>(As<ScalarType>(src->GetType()));

        // Load src to Vec tile if it is a global tensor
        ExprPtr src_tile = src;
        if (!src_is_scalar) {
          if (auto src_tensor_type = As<TensorType>(src->GetType())) {
            auto offsets = MakeZeroOffsetsTuple(src_tensor_type->shape_.size(), span);
            auto shapes = MakeShapesTuple(src_tensor_type->shape_, span);
            std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", MemorySpace::Vec},
                                                                     {"transpose", false}};
            auto load = op_reg.Create("tile.load", {src, offsets, shapes, shapes}, load_kw, span);
            auto src_var = std::make_shared<Var>("scatter_src", load->GetType(), span);
            prologue.push_back(std::make_shared<AssignStmt>(src_var, load, span));
            src_tile = src_var;
          }
        }

        CHECK(index_tile_type) << "tensor.scatter_: index must be a tile at this point";
        const DataType value_dtype = input_tile_type->dtype_;

        // Build nested ForStmt loops over each dimension of the index tile.
        // Recursion: build_loop(d, loop_vars) creates ForStmt for dimension d,
        // passing accumulated loop variables down to the body.
        std::function<StmtPtr(size_t, std::vector<VarPtr>&)> build_loop;
        build_loop = [&](size_t d, std::vector<VarPtr>& loop_vars) -> StmtPtr {
          if (d == rank) {
            // Innermost body: generate tile.read + tile.write statements
            std::vector<StmtPtr> body_stmts;

            auto bind = [&](const std::string& name, const ExprPtr& expr) -> ExprPtr {
              auto var = std::make_shared<Var>(name, expr->GetType(), span);
              body_stmts.push_back(std::make_shared<AssignStmt>(var, expr, span));
              return var;
            };

            // 1. Read index value: idx_val = tile.read(index, [loop_vars...])
            std::vector<ExprPtr> idx_elems;
            idx_elems.reserve(rank);
            for (size_t k = 0; k < rank; ++k) {
              idx_elems.push_back(loop_vars[k]);
            }
            auto read_indices = std::make_shared<MakeTuple>(idx_elems, span);
            auto idx_val = bind("scatter_idx_val", op_reg.Create("tile.read", {index_tile, read_indices}, {}, span));

            // 2. Read or get src value
            ExprPtr src_val;
            if (src_is_scalar) {
              src_val = src;
            } else {
              auto src_indices = std::make_shared<MakeTuple>(idx_elems, span);
              src_val = bind("scatter_src_val", op_reg.Create("tile.read", {src_tile, src_indices}, {}, span));
            }

            // Cast src_val to input dtype if needed
            if (As<ScalarType>(src_val->GetType())) {
              DataType src_dtype = GetScalarDtype(src_val);
              if (src_dtype != value_dtype) {
                src_val = bind("scatter_src_cast", MakeCast(src_val, value_dtype, span));
              }
            }

            // 3. Build write indices: same as loop_vars but dim-th replaced with idx_val
            std::vector<ExprPtr> write_idx_elems;
            write_idx_elems.reserve(rank);
            for (size_t k = 0; k < rank; ++k) {
              if (static_cast<int>(k) == dim) {
                write_idx_elems.push_back(idx_val);
              } else {
                write_idx_elems.push_back(loop_vars[k]);
              }
            }
            auto write_indices = std::make_shared<MakeTuple>(write_idx_elems, span);

            // 4. Write (with optional reduce)
            if (reduce == "none") {
              body_stmts.push_back(std::make_shared<EvalStmt>(
                  op_reg.Create("tile.write", {input, write_indices, src_val}, {}, span), span));
            } else {
              auto old_val = bind("scatter_old_val", op_reg.Create("tile.read", {input, write_indices}, {}, span));
              ExprPtr new_val;
              if (reduce == "add") {
                new_val = bind("scatter_new_val", MakeAdd(old_val, src_val, span));
              } else {
                new_val = bind("scatter_new_val", MakeMul(old_val, src_val, span));
              }
              body_stmts.push_back(std::make_shared<EvalStmt>(
                  op_reg.Create("tile.write", {input, write_indices, new_val}, {}, span), span));
            }

            // 5. Vector barrier after tile.write (required for tsetval hardware sync)
            body_stmts.push_back(std::make_shared<EvalStmt>(
                op_reg.Create("system.bar_v", {}, {}, span), span));

            return SeqStmts::Flatten(std::move(body_stmts), span);
          }

          // Create ForStmt for dimension d
          auto loop_var = std::make_shared<Var>(
              "scatter_d" + std::to_string(d), std::make_shared<ScalarType>(DataType::INDEX), span);
          loop_vars.push_back(loop_var);
          auto body = build_loop(d + 1, loop_vars);
          loop_vars.pop_back();

          auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
          auto step = std::make_shared<ConstInt>(1, DataType::INDEX, span);
          auto extent = index_tile_type->shape_[d];

          return std::make_shared<ForStmt>(
              loop_var, zero, extent, step,
              /*iter_args=*/std::vector<IterArgPtr>{},
              body,
              /*return_vars=*/std::vector<VarPtr>{},
              span, ForKind::Sequential);
        };

        std::vector<VarPtr> loop_vars;
        loop_vars.reserve(rank);
        auto nested_for = build_loop(0, loop_vars);

        // Emit bar_all before and after the scatter loops to ensure:
        // - Pre-loop: TLOAD (index) and TEXPANDS (input fill) have completed
        // - Post-loop: All tsetval writes are committed before tstore
        prologue.push_back(std::make_shared<EvalStmt>(
            op_reg.Create("system.bar_all", {}, {}, span), span));
        prologue.push_back(nested_for);
        prologue.push_back(std::make_shared<EvalStmt>(
            op_reg.Create("system.bar_all", {}, {}, span), span));

        // The input tile is modified in-place via tile.write; return it as the result.
        return ConversionResult{std::move(prologue), input};
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.create → tile.create
  //
  // tensor.create(shape, dtype=...) → tile.create(shape, dtype=..., target_memory=Vec)
  // If all shape dimensions are static constants, validate that the tile
  // fits within the target memory space (obtained from Backend::GetMemSize).
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.create",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 1) << "tensor.create conversion expects 1 arg (shape)";
        auto& op_reg = OpRegistry::GetInstance();

        MemorySpace target_mem = MemorySpace::Vec;
        std::vector<std::pair<std::string, std::any>> new_kwargs;
        for (const auto& [key, value] : kwargs) {
          if (key == "dtype") {
            new_kwargs.emplace_back(key, value);
          }
        }
        new_kwargs.emplace_back("target_memory", target_mem);

        // Static buffer size check when all shape dims are ConstInt
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
              CHECK(mem_size == 0 || tile_bytes <= mem_size)
                  << "tensor.create: tile size (" << tile_bytes << " bytes) exceeds buffer capacity ("
                  << mem_size << " bytes) for memory space " << static_cast<int>(target_mem) << " at "
                  << span.to_string();
            }
          }
        }

        auto create_call = op_reg.Create("tile.create", args, new_kwargs, span);
        return ConversionResult{create_call};
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.read → tensor.read (gm_tensor) or tile.read (local_tensor)
  //
  // gm_tensor.read(tensor, indices) stays as tensor.read (no conversion)
  // local_tensor.read(tile, indices) → tile.read(tile, indices)
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.read",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 2) << "tensor.read conversion expects 2 args (tensor, indices)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        if (As<TensorType>(input->GetType())) {
          // gm_tensor: keep as tensor.read
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tensor.read", args, span)};
          }
          return ConversionResult{op_reg.Create("tensor.read", args, kwargs, span)};
        }

        if (As<TileType>(input->GetType())) {
          // local_tensor (now tile): convert to tile.read
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tile.read", args, span)};
          }
          return ConversionResult{op_reg.Create("tile.read", args, kwargs, span)};
        }

        CHECK(false) << "tensor.read conversion: unexpected input type: " << input->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // ────────────────────────────────────────────────────────────────────────
  //
  // gm_tensor.write(tensor, indices, value) stays as tensor.write (no conversion)
  // local_tensor.write(tile, indices, value) → tile.write(tile, indices, value)
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.write",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.write conversion expects 3 args (tensor, indices, value)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& dest = args[0];

        if (As<TensorType>(dest->GetType())) {
          // gm_tensor: keep as tensor.write
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tensor.write", args, span)};
          }
          return ConversionResult{op_reg.Create("tensor.write", args, kwargs, span)};
        }

        if (As<TileType>(dest->GetType())) {
          // local_tensor (now tile): convert to tile.write
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tile.write", args, span)};
          }
          return ConversionResult{op_reg.Create("tile.write", args, kwargs, span)};
        }

        CHECK(false) << "tensor.write conversion: unexpected input type: " << dest->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });
}

void OpConversionRegistry::RegisterSimple(const std::string& from_op, const std::string& to_op) {
  // Capture to_op by value for the lambda
  conversions_[from_op] = [to_op](const std::vector<ExprPtr>& args,
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
}

void OpConversionRegistry::RegisterCustom(const std::string& from_op, ConversionFunc func) {
  conversions_[from_op] = std::move(func);
}

const ConversionFunc* OpConversionRegistry::Lookup(const std::string& op_name) const {
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
