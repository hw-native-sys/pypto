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

        // Extract pad_value kwarg (if any) to forward to the emitted tile.slice.
        std::vector<std::pair<std::string, std::any>> forward_kwargs;
        for (const auto& kv : kwargs) {
          if (kv.first == "pad_value") {
            forward_kwargs.push_back(kv);
            break;
          }
        }

        auto tensor_type = As<TensorType>(input->GetType());
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

  RegisterCustom(
      "tensor.expand_clone",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 2) << "tensor.expand_clone conversion expects 2 args (input, target)";

        auto& op_reg = OpRegistry::GetInstance();
        const auto& input = args[0];
        const auto& target = args[1];

        auto input_tensor_type = As<TensorType>(input->GetType());
        CHECK(input_tensor_type) << "tensor.expand_clone conversion: input must be TensorType, but got "
                                 << input->GetType()->TypeName();

        auto target_tensor_type = As<TensorType>(target->GetType());
        CHECK(target_tensor_type) << "tensor.expand_clone conversion: target must be TensorType, but got "
                                  << target->GetType()->TypeName();

        const auto& input_shape = input_tensor_type->shape_;
        const auto& target_shape = target_tensor_type->shape_;

        CHECK(input_shape.size() == 3)
            << "tensor.expand_clone conversion: input rank must be 3, but got " << input_shape.size();
        CHECK(target_shape.size() == input_shape.size())
            << "tensor.expand_clone conversion: input rank (" << input_shape.size()
            << ") must match target rank (" << target_shape.size() << ")";

        int broadcast_dim = -1;
        for (size_t i = 0; i < input_shape.size(); ++i) {
          if (DimensionsEqual(input_shape[i], target_shape[i])) {
            continue;
          }
          auto input_const = GetConstantDimension(input_shape[i]);
          CHECK(input_const && *input_const == 1)
              << "tensor.expand_clone conversion requires input dim " << i
              << " to be 1 for broadcasting, but got " << PythonPrint(input_shape[i]);
          CHECK(broadcast_dim < 0)
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

// ============================================================================
// Sort ops: sort32, mrgsort_format1, mrgsort_format2 — simple 1:1 name mapping.
// Auto-bridge in the convert pass loads TensorType args to Vec tiles.
// ============================================================================

void OpConversionRegistry::RegisterSortOps() {
  RegisterSimple("tensor.sort32", "tile.sort32");
  RegisterSimple("tensor.mrgsort_format1", "tile.mrgsort_format1");
  RegisterSimple("tensor.gather_mask", "tile.gather_mask");

  // tensor.mrgsort_format2: 2-4 srcs → tile.mrgsort_format2 with synthesized
  // scratch tmp + executed tiles allocated locally in Vec memory.
  //
  // The tile-level op requires (srcs..., tmp, executed). We don't expose tmp/
  // executed at the tensor level because:
  //   - tmp shape equals the merged output shape (sum of src last dims) — we
  //     can derive it; no user-visible value.
  //   - executed is a small [1, 4] INT16 hardware status tile that the PTO
  //     codegen actually materializes as a vector<4xi16> constant (the passed
  //     tile is a plumbing-only placeholder). Its shape can't round-trip
  //     through GM because 4 * 2 bytes = 8 bytes violates the 32-byte tile
  //     row alignment PTO enforces.
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
        CHECK(args.size() >= 2 && args.size() <= 4)
            << "tensor.mrgsort_format2 conversion expects 2-4 src args, got " << args.size();
        auto& op_reg = OpRegistry::GetInstance();

        // After the framework's input_reqs bridge, all srcs should be Vec tiles.
        std::vector<std::shared_ptr<const TileType>> src_tile_types;
        src_tile_types.reserve(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
          auto tt = As<TileType>(args[i]->GetType());
          CHECK(tt) << "tensor.mrgsort_format2 conversion expects bridged Vec tile at arg " << i;
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

        // Synthesize executed: tile.create([1, 4], dtype=INT16, target_memory=Vec).
        // PTO codegen ignores the actual tile content and emits a vector<4xi16>
        // constant; this tile is needed solely to satisfy the tile op's arity.
        std::vector<ExprPtr> exe_shape = {
            std::make_shared<ConstInt>(1, DataType::INDEX, span),
            std::make_shared<ConstInt>(4, DataType::INDEX, span),
        };
        auto exe_shape_tuple = std::make_shared<MakeTuple>(exe_shape, span);
        std::vector<std::pair<std::string, std::any>> exe_create_kwargs = {
            {"dtype", DataType(DataType::INT16)}, {"target_memory", MemorySpace::Vec}};
        auto exe_create = op_reg.Create("tile.create", {exe_shape_tuple}, exe_create_kwargs, span);
        auto exe_var = std::make_shared<Var>("mrgsort2_executed", exe_create->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(exe_var, exe_create, span));

        // Assemble tile.mrgsort_format2 call: (src0..srcN-1, tmp, executed) + kwargs.
        std::vector<ExprPtr> tile_args(args.begin(), args.end());
        tile_args.push_back(tmp_var);
        tile_args.push_back(exe_var);
        auto mrgsort_call = op_reg.Create("tile.mrgsort_format2", tile_args, kwargs, span);
        return ConversionResult{std::move(prologue), mrgsort_call};
      },
      std::move(mrgsort2_input_reqs));
}

// ============================================================================
// Gather op: tensor.gather -> per-row tile.load + tile.gather + tile.assemble
// loop over the outer (non-gather) dimension.
//
// MVP supports 2D input with dim == -1 (last axis). Semantics:
//     out[b, k] = input[b, index[b, k]]
//
// The loop yields a TileType accumulator built via tile.assemble; Phase 3 of
// ConvertTensorToTileOps then rewrites the loop to write directly into an
// added Out tensor parameter via tile.store (see RewriteReturnedAssembleLoop-
// ToStore). Using per-row slicing avoids materialising a `b * N` base offset
// tile, which would otherwise require an arange/iota op PyPTO currently lacks.
// ============================================================================

void OpConversionRegistry::RegisterGatherOps() {
  RegisterCustom(
      "tensor.gather",
      [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
         const Span& span) -> ConversionResult {
        CHECK(args.size() == 2) << "tensor.gather conversion expects 2 args (input, index), got "
                                << args.size();
        auto& op_reg = OpRegistry::GetInstance();

        const auto& input = args[0];
        const auto& index = args[1];

        auto input_tensor_type = As<TensorType>(input->GetType());
        auto input_tile_type = As<TileType>(input->GetType());
        CHECK(input_tensor_type || input_tile_type)
            << "tensor.gather conversion: input must be TensorType or TileType, got "
            << input->GetType()->TypeName();
        auto index_tensor_type = As<TensorType>(index->GetType());
        auto index_tile_type = As<TileType>(index->GetType());
        CHECK(index_tensor_type || index_tile_type)
            << "tensor.gather conversion: index must be TensorType or TileType, got "
            << index->GetType()->TypeName();
        const auto& input_shape = input_tensor_type ? input_tensor_type->shape_ : input_tile_type->shape_;
        const auto& index_shape = index_tensor_type ? index_tensor_type->shape_ : index_tile_type->shape_;
        CHECK(input_shape.size() == 2 && index_shape.size() == 2)
            << "tensor.gather conversion (MVP): only rank-2 inputs supported";

        int dim_val = GetKwargOr<int>(kwargs, "dim", -1);
        const int64_t rank = static_cast<int64_t>(input_shape.size());
        CHECK(dim_val == -1 || dim_val == rank - 1)
            << "tensor.gather conversion (MVP): only dim=-1 supported, got " << dim_val;

        DataType input_dtype = input_tensor_type ? input_tensor_type->dtype_ : input_tile_type->dtype_;

        auto make_index_const = [&](int64_t value) -> ExprPtr {
          return std::make_shared<ConstInt>(value, DataType::INDEX, span);
        };
        ExprPtr zero = make_index_const(0);
        ExprPtr one = make_index_const(1);

        std::vector<StmtPtr> prologue;

        // Accumulator tile, shape equal to the output tensor; Phase 3 later
        // rewrites this init to the added Out tensor parameter.
        auto acc_shape_tuple = MakeShapeTuple(index_shape, span);
        std::vector<std::pair<std::string, std::any>> acc_create_kwargs = {
            {"dtype", input_dtype}, {"target_memory", MemorySpace::Vec}};
        auto acc_create_call = op_reg.Create("tile.create", {acc_shape_tuple}, acc_create_kwargs, span);
        auto acc_init_var = std::make_shared<Var>("gather_out_init", acc_create_call->GetType(), span);
        prologue.push_back(std::make_shared<AssignStmt>(acc_init_var, acc_create_call, span));

        auto acc_tile_type = acc_create_call->GetType();
        auto loop_var = std::make_shared<Var>("b", std::make_shared<ScalarType>(DataType::INDEX), span);
        auto iter_arg = std::make_shared<IterArg>("gather_acc", acc_tile_type, acc_init_var, span);
        auto return_var = std::make_shared<Var>("gather_result", acc_tile_type, span);

        auto row_offsets = std::make_shared<MakeTuple>(std::vector<ExprPtr>{loop_var, zero}, span);
        std::vector<ExprPtr> src_row_shape = {one, input_shape[1]};
        std::vector<ExprPtr> idx_row_shape = {one, index_shape[1]};
        auto src_shape_tuple = MakeShapeTuple(src_row_shape, span);
        auto idx_shape_tuple = MakeShapeTuple(idx_row_shape, span);

        std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                     {"transpose", false}};

        auto make_row = [&](const ExprPtr& tensor_or_tile, bool is_tensor,
                            const ExprPtr& row_shape_tuple) -> CallPtr {
          if (is_tensor) {
            return op_reg.Create("tile.load", {tensor_or_tile, row_offsets, row_shape_tuple, row_shape_tuple},
                                 load_kwargs, span);
          }
          return op_reg.Create("tile.slice", {tensor_or_tile, row_shape_tuple, row_offsets, row_shape_tuple},
                               span);
        };

        std::vector<StmtPtr> body_stmts;
        auto src_load = make_row(input, input_tensor_type != nullptr, src_shape_tuple);
        auto src_tile_var = std::make_shared<Var>("gather_src_row", src_load->GetType(), span);
        body_stmts.push_back(std::make_shared<AssignStmt>(src_tile_var, src_load, span));

        auto idx_load = make_row(index, index_tensor_type != nullptr, idx_shape_tuple);
        auto idx_tile_var = std::make_shared<Var>("gather_idx_row", idx_load->GetType(), span);
        body_stmts.push_back(std::make_shared<AssignStmt>(idx_tile_var, idx_load, span));

        // Scratch workspace tile required by tile.gather (same shape as idx row, INT32).
        std::vector<std::pair<std::string, std::any>> tmp_create_kwargs = {
            {"dtype", DataType(DataType::INT32)}, {"target_memory", MemorySpace::Vec}};
        auto tmp_create_call = op_reg.Create("tile.create", {idx_shape_tuple}, tmp_create_kwargs, span);
        auto tmp_tile_var = std::make_shared<Var>("gather_tmp", tmp_create_call->GetType(), span);
        body_stmts.push_back(std::make_shared<AssignStmt>(tmp_tile_var, tmp_create_call, span));

        auto gather_call = op_reg.Create("tile.gather", {src_tile_var, idx_tile_var, tmp_tile_var}, span);
        auto gather_row_var = std::make_shared<Var>("gather_row", gather_call->GetType(), span);
        body_stmts.push_back(std::make_shared<AssignStmt>(gather_row_var, gather_call, span));

        // Assemble the per-row result into the accumulator; Phase 3 rewrites
        // this tile.assemble into tile.store once the Out param is added.
        auto assemble_call = op_reg.Create("tile.assemble", {iter_arg, gather_row_var, row_offsets}, span);
        auto assemble_var = std::make_shared<Var>("gather_assemble", assemble_call->GetType(), span);
        body_stmts.push_back(std::make_shared<AssignStmt>(assemble_var, assemble_call, span));
        body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{assemble_var}, span));

        auto body = SeqStmts::Flatten(std::move(body_stmts), span);
        auto for_stmt =
            std::make_shared<ForStmt>(loop_var, zero, input_shape[0], one, std::vector<IterArgPtr>{iter_arg},
                                      body, std::vector<VarPtr>{return_var}, span);
        prologue.push_back(for_stmt);
        return ConversionResult{std::move(prologue), return_var};
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
