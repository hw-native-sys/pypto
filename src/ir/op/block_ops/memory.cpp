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
 * @brief Memory block operations (get_block_idx, load, store)
 *
 * This file implements memory operations for block-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <any>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/common.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// ============================================================================
// CCE Codegen for block.load
// ============================================================================
CCECodegenFunc MakeBlockLoadCodegenCCE() {
  return [](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    CHECK(op->args_.size() == 5)
        << "block.load requires 5 arguments: tensor, row_offset, col_offset, height, width";

    auto src_tensor_var_ptr = std::dynamic_pointer_cast<const Var>(op->args_[0]);
    CHECK(src_tensor_var_ptr != nullptr) << "block.load source tensor must be a Var";

    std::string src_tensor_var = codegen.GetVarName(src_tensor_var_ptr);
    std::string row_offset = codegen.GetExprAsCode(op->args_[1]);
    std::string col_offset = codegen.GetExprAsCode(op->args_[2]);

    auto src_tensor_type = std::dynamic_pointer_cast<const TensorType>(src_tensor_var_ptr->GetType());
    CHECK(src_tensor_type != nullptr) << "block.load source must be TensorType";
    CHECK(src_tensor_type->shape_.size() >= 1) << "Tensor must be at least 1D";

    std::string stride_expr;
    if (src_tensor_type->shape_.size() == 1) {
      stride_expr = "1";
    } else {
      stride_expr = codegen.GetExprAsCode(src_tensor_type->shape_[src_tensor_type->shape_.size() - 1]);
    }

    std::string offset = row_offset + " * " + stride_expr + " + " + col_offset;
    std::string raw_ptr = codegen.GetPointer(src_tensor_var);
    std::string var_name = codegen.GetCurrentResultTarget();

    codegen.Emit("TASSIGN(" + src_tensor_var + ", " + raw_ptr + " + " + offset + ");");
    codegen.Emit("TLOAD(" + var_name + ", " + src_tensor_var + ");");
    return "";  // Statement-emitting mode
  };
}

// ============================================================================
// CCE Codegen for block.store
// ============================================================================
CCECodegenFunc MakeBlockStoreCodegenCCE() {
  return [](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    CHECK(op->args_.size() == 6)
        << "block.store requires 6 arguments: tile, row_offset, col_offset, height, width, output_tensor";

    std::string src_tile = codegen.GetExprAsCode(op->args_[0]);
    std::string row_offset = codegen.GetExprAsCode(op->args_[1]);
    std::string col_offset = codegen.GetExprAsCode(op->args_[2]);

    auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const Var>(op->args_[5]);
    CHECK(dst_tensor_var_ptr != nullptr) << "block.store destination tensor must be a Var";

    std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);

    auto dst_tensor_type = std::dynamic_pointer_cast<const TensorType>(dst_tensor_var_ptr->GetType());
    CHECK(dst_tensor_type != nullptr) << "block.store destination must be TensorType";
    CHECK(dst_tensor_type->shape_.size() >= 1) << "Tensor must be at least 1D";

    std::string stride_expr;
    if (dst_tensor_type->shape_.size() == 1) {
      stride_expr = "1";
    } else {
      stride_expr = codegen.GetExprAsCode(dst_tensor_type->shape_[dst_tensor_type->shape_.size() - 1]);
    }

    std::string offset = row_offset + " * " + stride_expr + " + " + col_offset;
    std::string raw_ptr = codegen.GetPointer(dst_tensor_var);
    std::string var_name = codegen.GetCurrentResultTarget();

    codegen.Emit("TASSIGN(" + dst_tensor_var + ", " + raw_ptr + " + " + offset + ");");
    codegen.Emit("TSTORE(" + dst_tensor_var + ", " + src_tile + ");");
    codegen.RegisterOutputPointer(var_name, dst_tensor_var);
    codegen.Emit("auto " + var_name + " = " + dst_tensor_var + ";");
    return "";  // Statement-emitting mode
  };
}

// ============================================================================
// PTO Codegen for block.load (subview + tload)
// ============================================================================
PTOCodegenFunc MakeBlockLoadCodegenPTO() {
  return [](const CallPtr& op, codegen::PTOCodegen& codegen) -> std::string {
    auto tensor = As<Var>(op->args_[0]);
    INTERNAL_CHECK(tensor) << "block.load first argument must be a Var";

    int64_t row_off = codegen.GetConstIntValue(op->args_[1]);
    int64_t col_off = codegen.GetConstIntValue(op->args_[2]);
    int64_t height = codegen.GetConstIntValue(op->args_[3]);
    int64_t width = codegen.GetConstIntValue(op->args_[4]);

    auto tensor_type = As<TensorType>(tensor->GetType());
    INTERNAL_CHECK(tensor_type) << "block.load tensor argument must have TensorType";

    std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
    std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
    std::string tile_buf = codegen.GetCurrentResultTarget();
    INTERNAL_CHECK(!tile_buf.empty()) << "block.load requires assignment target (tile_buf)";

    std::string tile_view = codegen.NewTemp();
    std::ostringstream subview_line;
    subview_line << tile_view << " = pto.subview " << tensor_view;
    subview_line << ", offsets = [" << codegen.GetIndexConstant(row_off) << ", ";
    subview_line << codegen.GetIndexConstant(col_off) << "]";
    subview_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
    subview_line << codegen.GetIndexConstant(width) << "]";
    subview_line << " : !pto.tensor_view<2x" << dtype_str << "> -> !pto.tile_view<";
    subview_line << height << "x" << width << "x" << dtype_str << ">";
    codegen.Emit(subview_line.str());

    std::ostringstream tload_line;
    tload_line << "pto.tload ins(" << tile_view;
    tload_line << " : !pto.tile_view<" << height << "x" << width << "x" << dtype_str << ">) outs(";
    tload_line << tile_buf << " : !pto.tile_buf<loc=ub, dtype=" << dtype_str;
    tload_line << ", rows=" << height << ", cols=" << width;
    tload_line << ", v_row=" << height << ", v_col=" << width;
    tload_line << ", blayout=row_major, slayout=none_box, fractal=512, pad=0>)";
    codegen.Emit(tload_line.str());
    return "";  // Multi-line emission
  };
}

// ============================================================================
// PTO Codegen for block.store
// ============================================================================
PTOCodegenFunc MakeBlockStoreCodegenPTO() {
  return [](const CallPtr& op, codegen::PTOCodegen& codegen) -> std::string {
    auto tile = As<Var>(op->args_[0]);
    INTERNAL_CHECK(tile) << "block.store first argument must be a Var";

    int64_t row_off = codegen.GetConstIntValue(op->args_[1]);
    int64_t col_off = codegen.GetConstIntValue(op->args_[2]);
    int64_t height = codegen.GetConstIntValue(op->args_[3]);
    int64_t width = codegen.GetConstIntValue(op->args_[4]);
    auto output_tensor = As<Var>(op->args_[5]);
    INTERNAL_CHECK(output_tensor) << "block.store output_tensor must be a Var";

    auto tensor_type = As<TensorType>(output_tensor->GetType());
    INTERNAL_CHECK(tensor_type) << "block.store output_tensor must have TensorType";

    std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
    std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
    std::string tile_buf = codegen.GetVarName(tile);
    std::string tile_view = codegen.NewTemp();

    std::ostringstream subview_line;
    subview_line << tile_view << " = pto.subview " << tensor_view;
    subview_line << ", offsets = [" << codegen.GetIndexConstant(row_off) << ", ";
    subview_line << codegen.GetIndexConstant(col_off) << "]";
    subview_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
    subview_line << codegen.GetIndexConstant(width) << "]";
    subview_line << " : !pto.tensor_view<2x" << dtype_str << "> -> !pto.tile_view<";
    subview_line << height << "x" << width << "x" << dtype_str << ">";
    codegen.Emit(subview_line.str());

    std::ostringstream tstore_line;
    tstore_line << "pto.tstore ins(" << tile_buf;
    tstore_line << " : !pto.tile_buf<loc=ub, dtype=" << dtype_str << ", rows=" << height;
    tstore_line << ", cols=" << width << ", v_row=" << height << ", v_col=" << width;
    tstore_line << ", blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(";
    tstore_line << tile_view << " : !pto.tile_view<" << height << "x" << width << "x" << dtype_str << ">)";
    codegen.Emit(tstore_line.str());
    return "";  // Multi-line emission
  };
}

// ============================================================================
// CCE Codegen for block.move
// ============================================================================
CCECodegenFunc MakeBlockMoveCodegenCCE() {
  return [](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    CHECK(op->args_.size() == 1) << "block.move requires 1 argument: source tile";

    std::string src_tile = codegen.GetExprAsCode(op->args_[0]);
    std::string dst_tile = codegen.GetCurrentResultTarget();

    // Get transpose attribute from kwargs
    bool transpose = false;
    for (const auto& [key, value] : op->kwargs_) {
      if (key == "transpose") {
        transpose = std::any_cast<bool>(value);
        break;
      }
    }

    // Emit TMOV instruction
    if (transpose) {
      codegen.Emit("TMOV(" + dst_tile + ", " + src_tile + ", true);  // transpose=true");
    } else {
      codegen.Emit("TMOV(" + dst_tile + ", " + src_tile + ");");
    }

    return "";  // Statement-emitting mode
  };
}

// ============================================================================
// CCE/PTO Codegen for block.alloc (no-op: allocation handled elsewhere)
// ============================================================================
CCECodegenFunc MakeBlockAllocCodegenCCE() {
  return [](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    (void)op;
    (void)codegen;
    return "";  // No C++ emission - MemRef/Tile setup handled in prologue
  };
}

PTOCodegenFunc MakeBlockAllocCodegenPTO() {
  return [](const CallPtr& op, codegen::PTOCodegen& codegen) -> std::string {
    (void)op;
    (void)codegen;
    return "";  // No MLIR emission - pto.alloc_tile generated from MemRefs in TileTypes
  };
}

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

TypePtr DeduceBlockGetBlockIdxType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_idx returns INT32 scalar
  return std::make_shared<ScalarType>(DataType::INT32);
}

TypePtr DeduceBlockLoadType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // load signature: (tensor, row_offset, col_offset, height, width)
  // We need at least the tensor argument
  CHECK(args.size() >= 1) << "The operator " << op_name << " requires at least 1 argument, but got "
                          << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // If we have shape arguments (height, width), use them to determine tile shape
  // Otherwise, we need to infer from context or use dynamic dimensions
  std::vector<ExprPtr> tile_shape;

  if (args.size() >= 5) {
    // We have height and width arguments (args[3] and args[4])
    // These should be scalar expressions that we can use as dimensions
    // For now, we'll use them directly as shape dimensions
    tile_shape.push_back(args[3]);
    tile_shape.push_back(args[4]);
  } else {
    // Use dynamic dimensions if shape is not provided
    // Create ConstInt expressions for dynamic dimensions
    auto dynamic_dim_height =
        std::make_shared<ConstInt>(static_cast<int>(kDynamicDim), DataType::INT32, Span::unknown());
    auto dynamic_dim_width =
        std::make_shared<ConstInt>(static_cast<int>(kDynamicDim), DataType::INT32, Span::unknown());
    tile_shape.push_back(dynamic_dim_height);
    tile_shape.push_back(dynamic_dim_width);
  }

  // Return TileType with same dtype as tensor
  return std::make_shared<TileType>(tile_shape, tensor_type->dtype_);
}

TypePtr DeduceBlockStoreType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // store signature: (tile, row_offset, col_offset, height, width, output_tensor)
  // We need at least the tile and output_tensor arguments
  CHECK(args.size() >= 2) << "The operator " << op_name << " requires at least 2 arguments, but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Last argument should be the output tensor
  auto output_tensor_type = As<TensorType>(args.back()->GetType());
  CHECK(output_tensor_type) << "The operator " << op_name
                            << " requires last argument to be a TensorType, but got "
                            << args.back()->GetType()->TypeName();

  // store returns the output tensor (same type)
  return output_tensor_type;
}

TypePtr DeduceBlockMoveType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // Validate args: expect exactly 1 argument (tile)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // Validate first argument is TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Extract transpose attribute (default: false)
  bool transpose = GetKwarg<bool>(kwargs, "transpose", false);

  // Extract and validate target_memory attribute (required)
  int target_memory = GetKwarg<int>(kwargs, "target_memory");

  // Determine output shape based on transpose flag
  const auto& input_shape = tile_type->shape_;
  std::vector<ExprPtr> output_shape;

  if (transpose && input_shape.size() == 2) {
    // Transpose: swap dimensions [H, W] -> [W, H]
    output_shape = {input_shape[1], input_shape[0]};
  } else {
    // No transpose: keep original shape
    output_shape = input_shape;
  }

  // Return TileType with computed shape and same dtype (no explicit MemRef)
  return std::make_shared<TileType>(output_shape, tile_type->dtype_);
}

TypePtr DeduceBlockAllocType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // alloc signature: (memory_space, addr, size, id)
  // Takes MemRef fields as arguments and returns MemRefType
  CHECK(args.size() == 4) << "The operator " << op_name << " requires exactly 4 arguments, but got "
                          << args.size();

  // Return MemRefType
  return GetMemRefType();
}

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("block.get_block_idx")
    .set_op_category("BlockOp")
    .set_description("Get the current block index")
    .set_pipe(PipeType::S)
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockGetBlockIdxType(args, kwargs, "block.get_block_idx");
    });

REGISTER_OP("block.load")
    .set_op_category("BlockOp")
    .set_description("Copy data from tensor to unified buffer (tile)")
    .set_pipe(PipeType::MTE2)
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("row_offset", "Row offset (scalar)")
    .add_argument("col_offset", "Column offset (scalar)")
    .add_argument("height", "Tile height (scalar)")
    .add_argument("width", "Tile width (scalar)")
    .set_attr<int>("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockLoadType(args, kwargs, "block.load");
    })
    .f_codegen_cce(MakeBlockLoadCodegenCCE())
    .f_codegen_pto(MakeBlockLoadCodegenPTO());

REGISTER_OP("block.store")
    .set_op_category("BlockOp")
    .set_description("Copy data from unified buffer (tile) to tensor")
    .set_pipe(PipeType::MTE3)
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("row_offset", "Row offset (scalar)")
    .add_argument("col_offset", "Column offset (scalar)")
    .add_argument("height", "Output height (scalar)")
    .add_argument("width", "Output width (scalar)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockStoreType(args, kwargs, "block.store");
    })
    .f_codegen_cce(MakeBlockStoreCodegenCCE())
    .f_codegen_pto(MakeBlockStoreCodegenPTO());

REGISTER_OP("block.l0c_store")
    .set_op_category("BlockOp")
    .set_description("Copy data from L0C tile to GM tensor")
    .set_pipe(PipeType::FIX)
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("row_offset", "Row offset (scalar)")
    .add_argument("col_offset", "Column offset (scalar)")
    .add_argument("height", "Output height (scalar)")
    .add_argument("width", "Output width (scalar)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockStoreType(args, kwargs, "block.l0c_store");
    })
    .f_codegen_cce(MakeBlockStoreCodegenCCE());

REGISTER_OP("block.move")
    .set_op_category("BlockOp")
    .set_description("Move tile to memory levels (UB/L1/L0A/L0B) with optional transpose")
    .set_pipe(PipeType::MTE1)
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<bool>("transpose")
    .set_attr<int>("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMoveType(args, kwargs, "block.move");
    })
    .f_codegen_cce(MakeBlockMoveCodegenCCE());

REGISTER_OP("block.alloc")
    .set_op_category("BlockOp")
    .set_description("Allocate memory for a MemRef object")
    .set_pipe(PipeType::V)
    .add_argument("memory_space", "Memory space (int enum value)")
    .add_argument("addr", "Starting address expression")
    .add_argument("size", "Size in bytes (scalar)")
    .add_argument("id", "MemRef ID (scalar)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockAllocType(args, kwargs, "block.alloc");
    })
    .f_codegen_cce(MakeBlockAllocCodegenCCE())
    .f_codegen_pto(MakeBlockAllocCodegenPTO());

}  // namespace ir
}  // namespace pypto
