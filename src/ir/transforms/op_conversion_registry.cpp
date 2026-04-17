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
#include <memory>
#include <string>
#include <unordered_map>
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
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"

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
  RegisterSimple("tensor.recip", "tile.recip");
  RegisterSimple("tensor.exp", "tile.exp");
  RegisterSimple("tensor.sqrt", "tile.sqrt");
  RegisterSimple("tensor.cast", "tile.cast");

  // tensor.rsqrt → tile.rsqrt (basic) or tile.rsqrt(src, tmp) (high-precision).
  // The tmp scratch tile is allocated via tile.create when high_precision=True.
  RegisterCustom(
      "tensor.rsqrt",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 1) << "tensor.rsqrt conversion expects 1 arg, got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        bool high_precision = GetKwargOr<bool>(kwargs, "high_precision", false);
        if (!high_precision) {
          return ConversionResult{op_reg.Create("tile.rsqrt", {input}, span)};
        }

        auto tile_type = As<TileType>(input->GetType());
        CHECK(tile_type) << "tensor.rsqrt conversion: input must be TileType after memory promotion, got "
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
  RegisterSimple("tensor.row_expand", "tile.row_expand");
  RegisterSimple("tensor.row_expand_add", "tile.row_expand_add");
  RegisterSimple("tensor.row_expand_sub", "tile.row_expand_sub");
  RegisterSimple("tensor.col_expand", "tile.col_expand");
  RegisterSimple("tensor.col_expand_sub", "tile.col_expand_sub");
  RegisterSimple("tensor.col_expand_div", "tile.col_expand_div");
  RegisterSimple("tensor.expands", "tile.expands");

  RegisterSimple("tensor.reshape", "tile.reshape");
  RegisterSimple("tensor.transpose", "tile.transpose");
  RegisterSimple("tensor.concat", "tile.concat");
  RegisterSimple("tensor.set_validshape", "tile.set_validshape");

  RegisterSimple("tensor.full", "tile.full");
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
  RegisterCustom("tensor.maximum", MakeBroadcastBinaryConv("tile.maximum", "tile.maximum"));
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
        CHECK(args.size() == 3 || args.size() == 4)
            << "tensor.slice conversion expects 3 or 4 args (tensor, shape, offset[, valid_shape])";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];
        const auto& shape = args[1];
        const auto& offset = args[2];

        auto tensor_type = As<TensorType>(input->GetType());
        auto tile_type = As<TileType>(input->GetType());

        if (tensor_type) {
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
          auto slice_call = op_reg.Create("tile.slice", slice_args, span);
          return ConversionResult{slice_call};
        }

        CHECK(false) << "tensor.slice conversion: unexpected input type: " << input->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // tensor.assemble → tile.store or tile.assemble depending on types
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
          auto store_call = op_reg.Create("tile.store", {source, offset, target}, span);
          return ConversionResult{store_call};
        }

        if (source_tile_type && target_tile_type) {
          auto assemble_call = op_reg.Create("tile.assemble", {target, source, offset}, span);
          return ConversionResult{assemble_call};
        }

        if (target_tile_type && !source_tile_type) {
          auto source_tensor_type = As<TensorType>(source->GetType());
          CHECK(source_tensor_type) << "tensor.assemble: source must be TensorType or TileType, but got "
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

  // tensor.scatter_update → tile.scatter_update (local) or passthrough (global)
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
          if (kwargs.empty()) {
            return ConversionResult{op_reg.Create("tensor.scatter_update", args, span)};
          }
          return ConversionResult{op_reg.Create("tensor.scatter_update", args, kwargs, span)};
        }

        CHECK(As<TileType>(input->GetType()))
            << "tensor.scatter_update: unexpected input type: " << input->GetType()->TypeName();

        std::vector<StmtPtr> prologue;

        ExprPtr index_tile = index;
        if (auto index_tensor_type = As<TensorType>(index->GetType())) {
          auto offsets = MakeZeroOffsets(index_tensor_type->shape_.size(), span);
          auto shapes = MakeShapeTuple(index_tensor_type->shape_, span);
          std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", MemorySpace::Vec},
                                                                   {"transpose", false}};
          auto load = op_reg.Create("tile.load", {index, offsets, shapes, shapes}, load_kw, span);
          auto idx_var = std::make_shared<Var>("scatter_idx", load->GetType(), span);
          prologue.push_back(std::make_shared<AssignStmt>(idx_var, load, span));
          index_tile = idx_var;
        }

        ExprPtr src_tile = src;
        if (auto src_tensor_type = As<TensorType>(src->GetType())) {
          auto offsets = MakeZeroOffsets(src_tensor_type->shape_.size(), span);
          auto shapes = MakeShapeTuple(src_tensor_type->shape_, span);
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

  // tensor.create → tile.create with static buffer size validation
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

  // tensor.fillpad → tile.fillpad (with auto-load for TensorType inputs)
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
  RegisterCustom(
      "tensor.read",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 2) << "tensor.read conversion expects 2 args (tensor, indices)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];

        if (As<TensorType>(input->GetType())) {
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

        CHECK(false) << "tensor.read conversion: unexpected input type: " << input->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });

  // tensor.write → tensor.write (gm_tensor) or tile.write (local_tensor)
  RegisterCustom(
      "tensor.write",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.write conversion expects 3 args (tensor, indices, value)";
        auto& op_reg = OpRegistry::GetInstance();
        const auto& dest = args[0];

        if (As<TensorType>(dest->GetType())) {
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

        CHECK(false) << "tensor.write conversion: unexpected input type: " << dest->GetType()->TypeName();
        return ConversionResult{nullptr};  // unreachable
      });
}

// ============================================================================
// Matmul ops: tensor.matmul / tensor.matmul_acc with Mat-space input_reqs
// ============================================================================

void OpConversionRegistry::RegisterMatmulOps() {
  RegisterCustom(
      "tensor.matmul",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 2) << "tensor.matmul conversion expects 2 args (lhs, rhs)";
        return ConversionResult{OpRegistry::GetInstance().Create("tile.matmul", {args[0], args[1]}, span)};
      },
      {{0, {MemorySpace::Mat, "a_trans"}}, {1, {MemorySpace::Mat, "b_trans"}}});

  RegisterCustom(
      "tensor.matmul_acc",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 3) << "tensor.matmul_acc conversion expects 3 args (acc, lhs, rhs)";
        return ConversionResult{
            OpRegistry::GetInstance().Create("tile.matmul_acc", {args[0], args[1], args[2]}, span)};
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
      CHECK(args.size() == 1) << tile_op << " conversion expects 1 arg (input tile)";
      auto& op_reg = OpRegistry::GetInstance();

      const auto& input = args[0];
      auto tile_type = As<TileType>(input->GetType());
      CHECK(tile_type) << tile_op << " conversion: input must be TileType, got "
                       << input->GetType()->TypeName();

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
