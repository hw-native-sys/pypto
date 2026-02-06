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

#include <string>

#include "pypto/backend/910B_CCE/backend_910b_cce.h"
#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
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

// block.load: emit TASSIGN + TLOAD (same format as original IR layer codegen)
// IR signature: (tensor, row_offset, col_offset, height, width) = 5 args
static std::string MakeBlockLoadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 5)
      << "block.load requires 5 arguments: tensor, row_offset, col_offset, height, width";

  auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tensor_var_ptr != nullptr) << "block.load source tensor must be a Var";

  std::string src_tensor_var = codegen.GetVarName(src_tensor_var_ptr);
  std::string row_offset = codegen.GetExprAsCode(op->args_[1]);
  std::string col_offset = codegen.GetExprAsCode(op->args_[2]);

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
// IR signature: (tile, row_offset, col_offset, height, width, output_tensor) = 6 args
static std::string MakeBlockStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 6)
      << "block.store requires 6 arguments: tile, row_offset, col_offset, height, width, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);
  std::string row_offset = codegen.GetExprAsCode(op->args_[1]);
  std::string col_offset = codegen.GetExprAsCode(op->args_[2]);

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[5]);
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
// IR signature: (tile, row_offset, col_offset, height, width, output_tensor) = 6 args
static std::string MakeBlockL0CStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 6)
      << "block.l0c_store requires 6 arguments: tile, row_offset, col_offset, height, width, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);
  std::string row_offset = codegen.GetExprAsCode(op->args_[1]);
  std::string col_offset = codegen.GetExprAsCode(op->args_[2]);

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[5]);
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
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

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
    codegen.Emit("TMOV(" + dst + ", " + src + ", true);  // transpose=true");
  } else {
    codegen.Emit("TMOV(" + dst + ", " + src + ");");
  }
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

      std::string acc = codegen.GetExprAsCode(op->args_[0]);
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst = codegen.GetCurrentResultTarget();

      // TMATMUL_ACC accumulates into dst, which should be initialized from acc
      codegen.Emit("TMATMUL_ACC(" + dst + ", " + lhs + ", " + rhs + ");");

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

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.alloc")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockAllocCodegenCCE(op, codegen);
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

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_block_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetBlockIdxCodegenCCE(op, codegen);
    });

// ============================================================================
// Reduction Operations
// ============================================================================

// Helper function for reduction operations (sum, max)
static std::string MakeBlockReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << op_prefix << " requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  int axis = op->GetKwarg<int>("axis");
  if (axis == 0) {
    codegen.Emit("TCOL" + op_prefix + "(" + dst + ", " + src + ");");
  } else {
    codegen.Emit("TROW" + op_prefix + "(" + dst + ", " + src + ");");
  }
  return "";
}

static std::string MakeBlockRowSumCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.row_sum requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit("TROWSUM(" + dst + ", " + src + ");");
  return "";
}

static std::string MakeBlockRowMaxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.row_max requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit("TROWMAX(" + dst + ", " + src + ");");
  return "";
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
      return MakeBlockRowSumCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowMaxCodegenCCE(op, codegen);
    });

// ============================================================================
// Broadcast Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TSUB", op, codegen);
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

REGISTER_BACKEND_OP(Backend910B_CCE, "block.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockTransformCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockTransformCodegenCCE(op, codegen);
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
