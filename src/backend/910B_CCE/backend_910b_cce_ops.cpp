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
 * This file registers all block operations for the CCE backend.
 * Each registration specifies the pipe type and CCE codegen function.
 */

#include <any>
#include <cstddef>
#include <memory>
#include <string>

#include "pypto/backend/910B_CCE/backend_910b_cce.h"
#include "pypto/backend/common/backend.h"
#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

// ============================================================================
// Helper Functions for CCE Code Generation
// ============================================================================

// Helper function for binary elementwise operations
static std::string MakeBinaryElementwiseCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Binary elementwise op requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// Helper function for binary scalar operations
static std::string MakeBinaryScalarCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Binary scalar op requires 2 arguments";
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

// Helper for block.cast - extract target_dtype from kwargs and use TCVT
static std::string MakeBlockCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.cast requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  int mode = op->GetKwarg<int>("mode");
  // TCVT signature: TCVT(dst, src, rmode)
  // Using default rounding mode (0 for round-to-nearest-even)
  codegen.Emit("TCVT(" + dst + ", " + src + ", " + std::to_string(mode) + ");");
  return "";
}

// Helper for block.cmp/cmps - extract cmp_type from kwargs and use TCMP
static std::string MakeBlockCmpCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                          codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.cmp requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  int cmp_type = op->GetKwarg<int>("cmp_type");
  // signature: TCMP/TCMPS(dst, src0, src1, cmpMode)
  // cmpMode: EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ", " + std::to_string(cmp_type) + ");");
  return "";
}

// Helper for block.expands/col_expand - expand scalar/col tile to tile
static std::string MakeBlockExpandsCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.expands/col_expand requires 2 arguments";

  std::string src1 = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  // FIX: this instruction is inplaced, dst and target addr should be same
  codegen.Emit(cce_op_name + "(" + dst + ", " + src1 + ");");
  return "";
}

// block.load: emit TASSIGN + TLOAD (same format as original IR layer codegen)
// IR signature: (tensor, offsets_tuple, shapes_tuple) = 3 args
static std::string MakeBlockLoadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "block.load requires 3 arguments: tensor, offsets, shapes";

  auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tensor_var_ptr != nullptr) << "block.load source tensor must be a Var";

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.load second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.load offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.load third argument must be a tuple (shapes)";

  std::string src_tensor_var = codegen.GetVarName(src_tensor_var_ptr);
  std::string row_offset = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  std::string col_offset =
      offsets_tuple->elements_.size() > 1 ? codegen.GetExprAsCode(offsets_tuple->elements_[1]) : "0";

  auto src_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(src_tensor_var_ptr->GetType());
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
  return "";
}

// block.store: emit TASSIGN + TSTORE + RegisterOutputPointer (same format as original IR layer codegen)
// IR signature: (tile, offsets_tuple, shapes_tuple, output_tensor) = 4 args
static std::string MakeBlockStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "block.store requires 4 arguments: tile, offsets, shapes, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.store second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.store offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.store third argument must be a tuple (shapes)";

  std::string row_offset = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  std::string col_offset =
      offsets_tuple->elements_.size() > 1 ? codegen.GetExprAsCode(offsets_tuple->elements_[1]) : "0";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[3]);
  CHECK(dst_tensor_var_ptr != nullptr) << "block.store destination tensor must be a Var";

  std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);

  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
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
  return "";
}

// Helper function for block.l0c_store
// IR signature: (tile, offsets_tuple, shapes_tuple, output_tensor) = 4 args
static std::string MakeBlockL0CStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4)
      << "block.l0c_store requires 4 arguments: tile, offsets, shapes, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.l0c_store second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.l0c_store offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.l0c_store third argument must be a tuple (shapes)";

  std::string row_offset = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  std::string col_offset =
      offsets_tuple->elements_.size() > 1 ? codegen.GetExprAsCode(offsets_tuple->elements_[1]) : "0";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[3]);
  CHECK(dst_tensor_var_ptr != nullptr) << "block.l0c_store destination tensor must be a Var";

  std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);

  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
  CHECK(dst_tensor_type != nullptr) << "block.l0c_store destination must be TensorType";
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
  return "";
}

// Helper function for block.move
static std::string MakeBlockMoveCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.move requires 1 argument: src";

  // Validate memory locations: can't UB→UB copies
  auto src_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(src_type != nullptr) << "Internal error: block.move source must be TileType";
  INTERNAL_CHECK(src_type->memref_.has_value())
      << "Internal error: block.move source TileType must have MemRef (InitMemRef pass should have run)";

  ir::MemorySpace target_memory = op->GetKwarg<ir::MemorySpace>("target_memory");
  ir::MemorySpace src_mem = src_type->memref_.value()->memory_space_;
  CHECK(!(src_mem == ir::MemorySpace::UB && target_memory == ir::MemorySpace::UB))
      << "block.move: UB to UB move should use block.ub_copy";

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

  codegen.Emit("TMOV(" + dst + ", " + src + ");");

  return "";
}

// Helper function for block.ub_copy (UB to UB copy only)
static std::string MakeBlockUbCopyCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.ub_copy requires 1 argument: src";

  // Validate memory locations: ONLY support UB→UB copies
  auto src_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(src_type != nullptr) << "Internal error: block.ub_copy source must be TileType";
  INTERNAL_CHECK(src_type->memref_.has_value())
      << "Internal error: block.ub_copy source TileType must have MemRef (InitMemRef pass should have run)";

  // Verify source is on UB
  ir::MemorySpace src_mem = src_type->memref_.value()->memory_space_;
  CHECK(src_mem == ir::MemorySpace::UB)
      << "block.ub_copy: source must be on UB memory, got " << ir::MemorySpaceToString(src_mem);

  // Get source and destination expressions
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

  // Emit TMOV instruction for UB→UB copy
  codegen.Emit("TMOV(" + dst + ", " + src + ");");

  return "";
}

// Helper function for block.alloc (no-op: allocation handled elsewhere)
static std::string MakeBlockAllocCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - MemRef/Tile setup handled in prologue
}

// Helper function for block.get_block_idx
static std::string MakeBlockGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 0) << "block.get_block_idx requires no arguments";
  std::string dst = codegen.GetCurrentResultTarget();

  // Get axis from kwargs
  int axis = -1;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "axis") {
      axis = std::any_cast<int>(value);
      break;
    }
  }
  CHECK(axis >= 0) << "block.get_block_idx requires 'axis' kwarg";

  codegen.Emit(dst + " = GET_BLOCK_IDX(" + std::to_string(axis) + ");");
  return "";
}

// Helper function for block.create_tile (no-op: allocation handled elsewhere)
static std::string MakeBlockCreateTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - Tile declaration handled in prologue
}

// Helper function for block.full
static std::string MakeBlockFullCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TEXPANDS(" + dst + ", " + scalar + ");");
  return "";
}

// ============================================================================
// Matmul Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 2) << "block.matmul requires 2 arguments: lhs, rhs";

      std::string lhs = codegen.GetExprAsCode(op->args_[0]);
      std::string rhs = codegen.GetExprAsCode(op->args_[1]);
      std::string dst = codegen.GetCurrentResultTarget();

      codegen.Emit("TMATMUL(" + dst + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul_acc")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 3) << "block.matmul_acc requires 3 arguments: acc, lhs, rhs";

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

REGISTER_BACKEND_OP(Backend910B_CCE, "block.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.minimum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TMULS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TADDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TDIVS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TSUBS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCmpCodegenCCE("TCMP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cmps")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCmpCodegenCCE("TCMPS", op, codegen);
    });

// ============================================================================
// Unary Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TEXP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TNEG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.recip")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRECIP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.rsqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TLOG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TABS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRELU", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCastCodegenCCE(op, codegen);
    });

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.alloc")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockAllocCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.create_tile")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCreateTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockLoadCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.l0c_store")
    .set_pipe(ir::PipeType::FIX)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockL0CStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.move")
    .set_pipe(ir::PipeType::MTE1)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockMoveCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.ub_copy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockUbCopyCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_block_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetBlockIdxCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockFullCodegenCCE(op, codegen);
    });

// ============================================================================
// Reduction Operations
// ============================================================================

static std::string MakeBlockRowReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "TROW" << op_prefix << " requires 2 arguments";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tmp_tile = codegen.GetExprAsCode(op->args_[1]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TROW" + op_prefix + "(" + result + ", " + tile + ", " + tmp_tile + ");");
  return "";
}

static std::string MakeBlockColReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "TCOL" << op_prefix << " requires 1 argument";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TCOL" + op_prefix + "(" + result + ", " + tile + ");");
  return "";
}

// Helper function for reduction operations (sum, max)
static std::string MakeBlockReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  int axis = op->GetKwarg<int>("axis");
  if (axis == 0) {
    return MakeBlockColReductionCodegenCCE(op_prefix, op, codegen_base);
  } else {
    return MakeBlockRowReductionCodegenCCE(op_prefix, op, codegen_base);
  }
}

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("MIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("MIN", op, codegen);
    });

// ============================================================================
// Broadcast Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.fillpad")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TFILLPAD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockExpandsCodegenCCE("TCOLEXPAND", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.expands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockExpandsCodegenCCE("TEXPANDS", op, codegen);
    });

// ============================================================================
// Transform Operations (view/reshape/transpose: same buffer, reinterpret)
// ============================================================================

static std::string MakeBlockTransformCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() >= 1) << "block view/reshape/transpose require at least 1 argument";
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

REGISTER_BACKEND_OP(Backend910B_CCE, "block.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.view")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockTransformCodegenCCE(op, codegen);
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
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("set_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_dst")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("wait_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_v")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_V);");
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_m")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_M);");
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_ALL);");
      return "";
    });

}  // namespace backend
}  // namespace pypto
