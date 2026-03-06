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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
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

// Check if a shape dimension is statically equal to 1
bool IsConstOne(const ExprPtr& expr) {
  if (auto c = As<ConstInt>(expr)) {
    return c->value_ == 1;
  }
  return false;
}

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
  auto& reg = OpRegistry::GetInstance();

  // ────────────────────────────────────────────────────────────────────────
  // Simple 1:1 conversions (tensor op → tile op, same args/kwargs)
  // ────────────────────────────────────────────────────────────────────────

  // Scalar ops
  RegisterSimple("tensor.add_scalar", "tile.adds");
  RegisterSimple("tensor.sub_scalar", "tile.subs");
  RegisterSimple("tensor.mul_scalar", "tile.muls");
  RegisterSimple("tensor.div_scalar", "tile.divs");

  // Memory ops (scalar read from GM tensor)
  RegisterSimple("tensor.read", "tile.getval");

  // Unary ops
  RegisterSimple("tensor.exp", "tile.exp");
  RegisterSimple("tensor.cast", "tile.cast");

  // Transform ops
  RegisterSimple("tensor.reshape", "tile.reshape");
  RegisterSimple("tensor.transpose", "tile.transpose");

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
  // tensor.view → tile.load
  //
  // tensor.view(tensor, shape, offset) → tile.load(tensor, offset, shape, shape, target_memory=Vec)
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.view",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.view conversion expects 3 args (tensor, shape, offset)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];
        const auto& shape = args[1];
        const auto& offset = args[2];

        auto tensor_type = As<TensorType>(input->GetType());
        auto tile_type = As<TileType>(input->GetType());

        if (tensor_type) {
          // Input is a tensor (e.g., function parameter) → direct tile.load with offset
          std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec}};
          auto load_call = op_reg.Create("tile.load", {input, offset, shape, shape}, load_kwargs, span);
          return ConversionResult{load_call};
        }

        if (tile_type) {
          // Input is a tile (e.g., from a prior tensor.matmul conversion).
          // Round-trip: store tile → temp tensor, then load the subview.
          std::vector<StmtPtr> prologue;

          // Create a temporary tensor with the tile's shape
          auto tensor_shape = MakeShapesTuple(tile_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", tile_type->dtype_}};
          auto create_call = op_reg.Create("tensor.create", {tensor_shape}, create_kwargs, span);
          auto tmp_tensor_var = std::make_shared<Var>("view_tmp_tensor", create_call->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(tmp_tensor_var, create_call, span));

          // Store tile to the temp tensor at offset [0,0]
          auto store_offsets = MakeZeroOffsetsTuple(tile_type->shape_.size(), span);
          auto store_shapes = MakeShapesTuple(tile_type->shape_, span);
          auto store_call =
              op_reg.Create("tile.store", {input, store_offsets, store_shapes, tmp_tensor_var}, span);
          auto stored_var = std::make_shared<Var>("view_tmp_stored", store_call->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(stored_var, store_call, span));

          // Load the subview from the stored tensor
          std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec}};
          auto load_call = op_reg.Create("tile.load", {stored_var, offset, shape, shape}, load_kwargs, span);
          return ConversionResult{std::move(prologue), load_call};
        }

        CHECK(false) << "tensor.view conversion: unexpected input type: " << input->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.matmul → tile.load(Mat) + tile.move(L0A/L0B) + tile.matmul + tile.l0c_store
  //
  // tensor.matmul(lhs, rhs, a_trans=False, b_trans=True, c_matrix_nz=False)
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.matmul",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 2) << "tensor.matmul conversion expects 2 args (lhs, rhs)";
        auto& op_reg = OpRegistry::GetInstance();

        bool b_trans = GetKwargOr<bool>(kwargs, "b_trans", false);

        const auto& lhs = args[0];
        const auto& rhs = args[1];

        std::vector<StmtPtr> prologue;

        auto lhs_tile_type = As<TileType>(lhs->GetType());
        auto rhs_tile_type = As<TileType>(rhs->GetType());
        auto lhs_tensor_type = As<TensorType>(lhs->GetType());
        auto rhs_tensor_type = As<TensorType>(rhs->GetType());

        ExprPtr lhs_mat, rhs_mat;

        // Load lhs to Mat if it's a tensor, or move from Vec if it's a tile
        if (lhs_tensor_type) {
          auto offsets = MakeZeroOffsetsTuple(lhs_tensor_type->shape_.size(), span);
          auto shapes = MakeShapesTuple(lhs_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> kw = {{"target_memory", MemorySpace::Mat}};
          auto load = op_reg.Create("tile.load", {lhs, offsets, shapes, shapes}, kw, span);
          auto load_var = std::make_shared<Var>("lhs_mat", load->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(load_var, load, span));
          lhs_mat = load_var;
        } else if (lhs_tile_type) {
          std::vector<std::pair<std::string, std::any>> kw = {{"target_memory", MemorySpace::Mat},
                                                              {"transpose", false}};
          auto move = op_reg.Create("tile.move", {lhs}, kw, span);
          auto move_var = std::make_shared<Var>("lhs_mat", move->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(move_var, move, span));
          lhs_mat = move_var;
        } else {
          CHECK(false) << "tensor.matmul: unexpected lhs type: " << lhs->GetType()->TypeName();
        }

        if (rhs_tensor_type) {
          auto offsets = MakeZeroOffsetsTuple(rhs_tensor_type->shape_.size(), span);
          auto shapes = MakeShapesTuple(rhs_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> kw = {{"target_memory", MemorySpace::Mat}};
          auto load = op_reg.Create("tile.load", {rhs, offsets, shapes, shapes}, kw, span);
          auto load_var = std::make_shared<Var>("rhs_mat", load->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(load_var, load, span));
          rhs_mat = load_var;
        } else if (rhs_tile_type) {
          std::vector<std::pair<std::string, std::any>> kw = {{"target_memory", MemorySpace::Mat},
                                                              {"transpose", false}};
          auto move = op_reg.Create("tile.move", {rhs}, kw, span);
          auto move_var = std::make_shared<Var>("rhs_mat", move->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(move_var, move, span));
          rhs_mat = move_var;
        } else {
          CHECK(false) << "tensor.matmul: unexpected rhs type: " << rhs->GetType()->TypeName();
        }

        // Move lhs to L0A (Left)
        {
          std::vector<std::pair<std::string, std::any>> kw = {{"target_memory", MemorySpace::Left},
                                                              {"transpose", false}};
          auto move = op_reg.Create("tile.move", {lhs_mat}, kw, span);
          auto var = std::make_shared<Var>("lhs_l0a", move->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(var, move, span));
          lhs_mat = var;
        }

        // Move rhs to L0B (Right) with optional transpose
        {
          std::vector<std::pair<std::string, std::any>> kw = {{"target_memory", MemorySpace::Right},
                                                              {"transpose", b_trans}};
          auto move = op_reg.Create("tile.move", {rhs_mat}, kw, span);
          auto var = std::make_shared<Var>("rhs_l0b", move->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(var, move, span));
          rhs_mat = var;
        }

        auto matmul_call = op_reg.Create("tile.matmul", {lhs_mat, rhs_mat}, span);
        return ConversionResult{std::move(prologue), matmul_call};
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
      auto create_call = op_reg.Create("tile.create_tile", {shape_tuple}, create_kwargs, span);

      auto tmp_var = std::make_shared<Var>("tmp_tile", create_call->GetType(), span);
      std::vector<StmtPtr> prologue;
      prologue.push_back(std::make_shared<AssignStmt>(tmp_var, create_call, span));

      auto reduction_call = op_reg.Create(tile_op, {input, tmp_var}, span);
      return ConversionResult{std::move(prologue), reduction_call};
    };
  };

  RegisterCustom("tensor.row_max", MakeReductionConv("tile.row_max"));
  RegisterCustom("tensor.row_sum", MakeReductionConv("tile.row_sum"));

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
          // Both are tiles (e.g. target from tile.create_tile, source from conversion).
          // Store source into a temp tensor, then return the tensor.
          std::vector<StmtPtr> prologue;

          // Create a temporary tensor for the target
          auto target_shape = MakeShapesTuple(target_tile_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", target_tile_type->dtype_}};
          auto create_call = op_reg.Create("tensor.create", {target_shape}, create_kwargs, span);
          auto tmp_tensor = std::make_shared<Var>("assemble_tmp", create_call->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(tmp_tensor, create_call, span));

          // Store source tile into the tensor at the given offset
          auto store_call = op_reg.Create("tile.store", {source, offset, tmp_tensor}, span);
          return ConversionResult{std::move(prologue), store_call};
        }

        // Both still tensors — keep as tensor.assemble
        if (kwargs.empty()) {
          return ConversionResult{op_reg.Create("tensor.assemble", args, span)};
        }
        return ConversionResult{op_reg.Create("tensor.assemble", args, kwargs, span)};
      });

  // ────────────────────────────────────────────────────────────────────────
  // tensor.create → tile.create_tile
  //
  // tensor.create(shape, dtype=...) → tile.create_tile(shape, dtype=..., target_memory=Vec)
  // ────────────────────────────────────────────────────────────────────────

  RegisterCustom(
      "tensor.create",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 1) << "tensor.create conversion expects 1 arg (shape)";
        auto& op_reg = OpRegistry::GetInstance();

        auto new_kwargs = kwargs;
        new_kwargs.emplace_back("target_memory", MemorySpace::Vec);
        auto create_call = op_reg.Create("tile.create_tile", args, new_kwargs, span);
        return ConversionResult{create_call};
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
