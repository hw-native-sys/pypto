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
 * @file backend_910b_cce_ops.cpp
 * @brief Backend op registration for Backend910B_CCE
 *
 * This file registers all tile operations for the CCE backend.
 * Each registration specifies the pipe type and CCE codegen function.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "pypto/backend/910B_CCE/backend_910b_cce.h"
#include "pypto/backend/common/backend.h"
#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

// ============================================================================
// Helper Functions for CCE Code Generation
// ============================================================================

/**
 * @brief Compute stride-based offset for multi-dimensional tensor access
 * @param codegen The CCE codegen instance
 * @param tensor_struct The TensorData struct pointer name
 * @param offsets Tuple of offset expressions for each dimension
 * @param transpose DN load for matmul transpose
 * @return C++ expression string for total offset computation
 */
static std::string ComputeStrideBasedOffset(codegen::CCECodegen& codegen, const std::string& tensor_struct,
                                            const ir::MakeTuplePtr& offsets, bool transpose = false) {
  // Build offset computation: offset[0] * compute_stride(dim=0) + offset[1] * compute_stride(dim=1) + ...
  // Note: start_offset is already added to the base pointer, so we don't add it here
  std::ostringstream offset_computation;
  offset_computation << "(0";

  auto offset_elems = offsets->elements_;
  if (transpose) {
    INTERNAL_CHECK(offset_elems.size() >= 2)
        << "Internal error: transpose requires at least 2 offset dimensions, got " << offset_elems.size();
    std::iter_swap(offset_elems.rbegin(), offset_elems.rbegin() + 1);
  }
  for (size_t i = 0; i < offsets->elements_.size(); ++i) {
    offset_computation << " + " << codegen.GetExprAsCode(offset_elems[i]) << " * compute_stride("
                       << tensor_struct << ", " << i << ")";
  }

  offset_computation << ")";
  return offset_computation.str();
}

// Helper function for binary operations (elementwise and scalar)
static std::string MakeBinaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Binary op requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// Helper function for unary operations
static std::string MakeUnaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Unary op requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + src + ");");
  return "";
}

// Helper for tile.cast - extract target_dtype from kwargs and use TCVT
static std::string MakeTileCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "tile.cast requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  int mode = op->GetKwarg<int>("mode");
  // TCVT signature: TCVT(dst, src, rmode)
  // Using default rounding mode (0 for round-to-nearest-even)
  codegen.Emit("TCVT(" + dst + ", " + src + ", " + codegen.GetTypeConverter().ConvertCastRoundMode(mode) +
               ");");
  return "";
}

// Helper for tile.cmp/cmps - extract cmp_type from kwargs and use TCMP
static std::string MakeTileCmpCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.cmp requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  int cmp_type = op->GetKwarg<int>("cmp_type");
  // signature: TCMP/TCMPS(dst, src0, src1, cmpMode)
  // cmpMode: EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ", " + std::to_string(cmp_type) + ");");
  return "";
}

// Helper for tile.expands/col_expand - expand scalar/col tile to tile
static std::string MakeTileExpandsCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                             codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.expands/col_expand requires 2 arguments";

  std::string src1 = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  // FIX: this instruction is inplaced, dst and target addr should be same
  codegen.Emit(cce_op_name + "(" + dst + ", " + src1 + ");");
  return "";
}

// tile.load: generate temp GlobalTensor + TASSIGN + TLOAD
// IR signature: (tensor, offsets_tuple, shapes_tuple, validshape) = 4 args
static std::string MakeTileLoadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "tile.load requires 4 arguments: tensor, offsets, shapes, validshape";

  auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tensor_var_ptr != nullptr) << "tile.load source tensor must be a Var";

  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "tile.load second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "tile.load offsets tuple must have at least 1 element";

  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "tile.load third argument must be a tuple (shapes)";

  bool transpose = op->GetKwarg<bool>("transpose", false);

  auto src_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(src_tensor_var_ptr->GetType());
  CHECK(src_tensor_type != nullptr) << "tile.load source must be TensorType";

  // Look up pointer/struct by IR var name
  std::string src_ptr = codegen.GetPointer(src_tensor_var_ptr->name_);
  std::string src_struct = codegen.GetTensorStruct(src_tensor_var_ptr->name_);

  // Generate temp GlobalTensor with unique name and load shape
  int id = codegen.GetNextGlobalTensorId();
  std::string gm_name = "tmp_gm_" + std::to_string(id);
  codegen.GenerateGlobalTensorTypeDeclaration(gm_name, src_tensor_type, src_ptr, src_struct,
                                              shapes_tuple->elements_);

  // Compute stride-based offset and emit load
  std::string offset = ComputeStrideBasedOffset(codegen, src_struct, offsets_tuple, transpose);
  std::string var_name = codegen.GetCurrentResultTarget();

  codegen.Emit("TASSIGN(" + gm_name + ", " + src_ptr + " + " + offset + ");");
  codegen.Emit("TLOAD(" + var_name + ", " + gm_name + ");");
  return "";
}

// tile.store: generate temp GlobalTensor + TASSIGN + TSTORE + propagate pointer/struct
// IR signature: (tile, offsets_tuple, output_tensor) = 3 args
static std::string MakeTileStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.store requires 3 arguments: tile, offsets, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "tile.store second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "tile.store offsets tuple must have at least 1 element";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[2]);
  CHECK(dst_tensor_var_ptr != nullptr) << "tile.store destination tensor must be a Var";

  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
  CHECK(dst_tensor_type != nullptr) << "tile.store destination must be TensorType";

  // Look up pointer/struct by IR var name
  std::string dst_ptr = codegen.GetPointer(dst_tensor_var_ptr->name_);
  std::string dst_struct = codegen.GetTensorStruct(dst_tensor_var_ptr->name_);

  // Extract shape from source tile type
  auto src_tile_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tile_var != nullptr) << "block.store source must be a Var";
  auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(src_tile_var->GetType());
  CHECK(tile_type != nullptr) << "block.store source must have TileType";

  // Generate temp GlobalTensor with unique name and store shape
  int id = codegen.GetNextGlobalTensorId();
  std::string gm_name = "tmp_gm_" + std::to_string(id);
  codegen.GenerateGlobalTensorTypeDeclaration(gm_name, dst_tensor_type, dst_ptr, dst_struct,
                                              tile_type->shape_);

  // Compute stride-based offset and emit store
  std::string offset = ComputeStrideBasedOffset(codegen, dst_struct, offsets_tuple);
  std::string var_name = codegen.GetCurrentResultTarget();

  codegen.Emit("TASSIGN(" + gm_name + ", " + dst_ptr + " + " + offset + ");");
  codegen.Emit("TSTORE(" + gm_name + ", " + src_tile + ");");

  // Propagate raw pointer/struct to result var (for SSA chain)
  codegen.RegisterOutputPointer(var_name, dst_ptr);
  codegen.RegisterOutputTensorStruct(var_name, dst_struct);
  return "";
}

// Helper function for tile.move
static std::string MakeTileMoveCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "tile.move requires 1 argument: src";

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

  codegen.Emit("TMOV(" + dst + ", " + src + ");");

  return "";
}

// Helper function for tile.alloc (no-op: allocation handled elsewhere)
static std::string MakeTileAllocCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - MemRef/Tile setup handled in prologue
}

// Helper function for tile.get_block_idx
static std::string MakeTileGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 0) << "tile.get_block_idx requires no arguments";
  std::string dst = codegen.GetCurrentResultTarget();

  // Get axis from kwargs
  int axis = -1;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "axis") {
      axis = std::any_cast<int>(value);
      break;
    }
  }
  CHECK(axis >= 0) << "tile.get_block_idx requires 'axis' kwarg";

  codegen.Emit(dst + " = GET_BLOCK_IDX(" + std::to_string(axis) + ");");
  return "";
}

// Helper function for tile.create (no-op: allocation handled elsewhere)
static std::string MakeTileCreateTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - Tile declaration handled in prologue
}

// Helper function for tile.full
static std::string MakeTileFullCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TEXPANDS(" + dst + ", " + scalar + ");");
  return "";
}

// ============================================================================
// Matmul Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.matmul")
    .f_infer_pipe([](const ir::CallPtr&) -> ir::PipeType { return ir::PipeType::M; })
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 2) << "tile.matmul requires 2 arguments: lhs, rhs";

      std::string lhs = codegen.GetExprAsCode(op->args_[0]);
      std::string rhs = codegen.GetExprAsCode(op->args_[1]);
      std::string dst = codegen.GetCurrentResultTarget();

      codegen.Emit("TMATMUL(" + dst + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.matmul_acc")
    .f_infer_pipe([](const ir::CallPtr&) -> ir::PipeType { return ir::PipeType::M; })
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 3) << "tile.matmul_acc requires 3 arguments: acc, lhs, rhs";

      [[maybe_unused]] std::string acc = codegen.GetExprAsCode(op->args_[0]);
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst = codegen.GetCurrentResultTarget();

      // TMATMUL_ACC accumulates into dst, which should be initialized from acc
      // In CCE ISA, this is typically: TMATMUL_ACC(dst, acc, lhs, rhs)
      codegen.Emit("TMATMUL_ACC(" + dst + ", " + acc + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

// ============================================================================
// Elementwise Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.mul")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.add")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.div")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.sub")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.maximum")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.minimum")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TMIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.muls")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TMULS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.adds")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TADDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.divs")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TDIVS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.subs")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TSUBS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.cmp")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCmpCodegenCCE("TCMP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.cmps")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCmpCodegenCCE("TCMPS", op, codegen);
    });

// ============================================================================
// Unary Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.exp")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TEXP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.neg")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TNEG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.recip")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRECIP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.rsqrt")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.sqrt")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.log")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TLOG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.abs")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TABS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.relu")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRELU", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.cast")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCastCodegenCCE(op, codegen);
    });

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.alloc")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileAllocCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.create")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCreateTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.load")
    .f_infer_pipe([](const ir::CallPtr&) -> ir::PipeType { return ir::PipeType::MTE2; })
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileLoadCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.store")
    .f_infer_pipe([](const ir::CallPtr& call) -> ir::PipeType {
      CHECK(call != nullptr) << "tile.store infer_pipe received null call";
      CHECK(call->args_.size() == 3) << "tile.store requires 3 arguments";
      auto src_type = ir::As<ir::TileType>(call->args_[0]->GetType());
      if (src_type && src_type->memref_.has_value() && (*src_type->memref_ != nullptr)) {
        auto src_mem = (*src_type->memref_)->memory_space_;
        if (src_mem == ir::MemorySpace::Acc) {
          return ir::PipeType::FIX;
        }
      }
      return ir::PipeType::MTE3;
    })
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.move")
    .f_infer_pipe([](const ir::CallPtr& call) -> ir::PipeType {
      CHECK(call != nullptr) << "tile.move infer_pipe received null call";
      CHECK(call->args_.size() == 1) << "tile.move requires 1 argument";
      auto src_type = ir::As<ir::TileType>(call->args_[0]->GetType());
      if (src_type && src_type->memref_.has_value() && (*src_type->memref_ != nullptr)) {
        auto src_mem = (*src_type->memref_)->memory_space_;
        auto target_memory = call->GetKwarg<ir::MemorySpace>("target_memory");
        if (src_mem == ir::MemorySpace::Vec && target_memory == ir::MemorySpace::Vec) {
          return ir::PipeType::V;
        }
      }
      return ir::PipeType::MTE1;
    })
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileMoveCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.get_block_idx")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileGetBlockIdxCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.full")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileFullCodegenCCE(op, codegen);
    });

// ============================================================================
// Reduction Operations
// ============================================================================

static std::string MakeTileRowReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                  codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "TROW" << op_prefix << " requires 2 arguments";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tmp_tile = codegen.GetExprAsCode(op->args_[1]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TROW" + op_prefix + "(" + result + ", " + tile + ", " + tmp_tile + ");");
  return "";
}

static std::string MakeTileColReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                  codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "TCOL" << op_prefix << " requires 1 argument";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TCOL" + op_prefix + "(" + result + ", " + tile + ");");
  return "";
}

// Helper function for reduction operations (sum, max)
static std::string MakeTileReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  int axis = op->GetKwarg<int>("axis");
  if (axis == 0) {
    return MakeTileColReductionCodegenCCE(op_prefix, op, codegen_base);
  } else {
    return MakeTileRowReductionCodegenCCE(op_prefix, op, codegen_base);
  }
}

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.sum")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.max")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.row_sum")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileRowReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.row_max")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileRowReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.min")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReductionCodegenCCE("MIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.row_min")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileRowReductionCodegenCCE("MIN", op, codegen);
    });

// ============================================================================
// Broadcast Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.row_expand_mul")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TROWEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.row_expand_div")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TROWEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.row_expand_sub")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TROWEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.row_expand_add")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TROWEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.fillpad")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TFILLPAD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.col_expand")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileExpandsCodegenCCE("TCOLEXPAND", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.col_expand_add")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TCOLEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.col_expand_mul")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TCOLEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.col_expand_div")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TCOLEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.col_expand_sub")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenCCE("TCOLEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.expands")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileExpandsCodegenCCE("TEXPANDS", op, codegen);
    });

// ============================================================================
// Transform Operations (view/reshape/transpose: same buffer, reinterpret)
// ============================================================================

static std::string MakeTileTransformCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() >= 1) << "tile view/reshape/transpose require at least 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit("TMOV(" + dst + ", " + src + ");");
  return "";
}

static std::string MakeTileReshapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  std::string input_var = codegen.GetExprAsCode(op->args_[0]);

  codegen.Emit("TRESHAPE(" + target_var + ", " + input_var + ");");
  return "";
}

static std::string MakeTileTransposeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  std::string input_var = codegen.GetExprAsCode(op->args_[0]);
  auto axis1 = codegen.GetConstIntValue(op->args_[1]);
  auto axis2 = codegen.GetConstIntValue(op->args_[2]);
  size_t ndim = ir::As<ir::TileType>(op->args_[0]->GetType())->shape_.size();

  INTERNAL_CHECK(ndim == 2) << "Codegen only supports 2D tiles, but got " << ndim << "D tile";
  INTERNAL_CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=axis2="
                                 << axis1;
  INTERNAL_CHECK(axis1 >= 0 && axis1 < ndim && axis2 >= 0 && axis2 < ndim)
      << "tile.transpose: axis1 and axis2 must be in range [0, " << ndim << "), but got axis1=" << axis1
      << ", axis2=" << axis2;

  codegen.Emit("TTRANS(" + target_var + ", " + input_var + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.reshape")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.transpose")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileTransposeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "tile.slice")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileTransformCodegenCCE(op, codegen);
    });

// ============================================================================
// Sync / Barrier Operations (inserted by insert_sync_pass)
// ============================================================================

static std::string PipeTypeToCCEString(ir::PipeType pipe) {
  switch (pipe) {
    case ir::PipeType::MTE1:
      return "PIPE_MTE1";
    case ir::PipeType::MTE2:
      return "PIPE_MTE2";
    case ir::PipeType::MTE3:
      return "PIPE_MTE3";
    case ir::PipeType::M:
      return "PIPE_M";
    case ir::PipeType::V:
      return "PIPE_V";
    case ir::PipeType::S:
      return "PIPE_S";
    case ir::PipeType::FIX:
      return "PIPE_FIX";
    case ir::PipeType::ALL:
      return "PIPE_ALL";
    default:
      return "PIPE_V";
  }
}

static std::string MakeSyncCodegenCCE(const std::string& isa_name, const ir::CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
  auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
  int event_id = op->GetKwarg<int>("event_id");
  std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
  std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
  std::string event_id_str = "EVENT_ID" + std::to_string(event_id);
  codegen.Emit(isa_name + "(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id_str + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_src")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("set_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_dst")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("wait_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_v")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_V);");
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_m")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_M);");
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_all")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_ALL);");
      return "";
    });

static std::string MakeTensorDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  int64_t axis = codegen.GetConstIntValue(op->args_[1]);

  auto input_tensor = ir::As<ir::TensorType>(op->args_[0]->GetType());
  CHECK(input_tensor) << "tensor.dim need TensorType for first arg, but got "
                      << op->args_[0]->GetType()->TypeName();
  auto ndims = static_cast<int64_t>(input_tensor->shape_.size());
  int64_t pad_dims = 5 - ndims;  // pto-isa pad shape to 5 dims

  // get axis in GlobalTensor 5 dims
  if (axis < 0) {
    axis += ndims;
  }
  int64_t gt_dim = pad_dims + axis;

  // get GlobalTensor of input_tensor
  auto input_tensor_var = ir::As<ir::Var>(op->args_[0]);
  CHECK(input_tensor_var) << "tensor.dim need var with TensorType for first arg";
  std::string input_tensor_var_name = codegen.GetVarName(input_tensor_var);

  codegen.Emit("int " + target_var + " = " + input_tensor_var_name + ".GetShape(GlobalTensorDim::DIM_" +
               std::to_string(gt_dim) + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "tensor.dim")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorDimCodegenCCE(op, codegen);
    });

}  // namespace backend
}  // namespace pypto
