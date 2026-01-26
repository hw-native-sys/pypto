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

#include "pypto/codegen/code_generator.h"

#include <sstream>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {

namespace codegen {

CodeGenerator::CodeGenerator() = default;

std::string CodeGenerator::Generate(const ir::FunctionPtr& func) {
  CHECK(func != nullptr) << "Cannot generate code for null function";

  // Clear state
  emitter_.Clear();
  context_.Clear();

  // Generate prologue and body
  GeneratePrologue(func);
  GenerateBody(func);

  return emitter_.GetCode();
}

void CodeGenerator::GeneratePrologue(const ir::FunctionPtr& func) {
  // Function signature
  emitter_.EmitLine("__aicore__ __attribute__((always_inline)) void run" + func->name_ +
                     "(__gm__ int64_t* args)");
  emitter_.EmitLine("{");
  emitter_.IncreaseIndent();

  emitter_.EmitLine("// Unpack arguments");

  // First pass: Unpack tensor arguments (use sanitized names but don't register yet)
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    const std::string param_name = context_.SanitizeName(param);

    // Get tensor type information
    auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType());
    if (!tensor_type) {
      throw pypto::ValueError("Parameter " + param->name_ + " must have TensorType");
    }

    // Extract element type
    std::string element_type = type_converter_.ConvertDataType(tensor_type->dtype_);

    // Emit argument unpacking
    std::string unpacking_line = "__gm__ ";
    unpacking_line += element_type;
    unpacking_line += "* ";
    unpacking_line += param_name;
    unpacking_line += " = reinterpret_cast<__gm__ ";
    unpacking_line += element_type;
    unpacking_line += "*>(args[";
    unpacking_line += std::to_string(i);
    unpacking_line += "]);";
    emitter_.EmitLine(unpacking_line);
  }

  emitter_.EmitLine("");
  emitter_.EmitLine("// Global tensor declarations");

  // Second pass: Generate GlobalTensor type definitions and register with Global suffix
  for (const auto& param : func->params_) {
    const std::string base_name = context_.SanitizeName(param);

    // Register parameter with "Global" suffix for use in operations
    const std::string global_name = base_name + "Global";
    context_.RegisterVar(param, global_name);

    // Get tensor type information
    auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType());
    if (!tensor_type) {
      throw pypto::ValueError("Parameter " + param->name_ + " must have TensorType");
    }

    // Extract shape dimensions as constant integers
    std::vector<int64_t> shape_dims;
    shape_dims.reserve(tensor_type->shape_.size());
    for (const auto& dim_expr : tensor_type->shape_) {
      shape_dims.push_back(ExtractConstInt(dim_expr));
    }

    // Get element type
    std::string element_type = type_converter_.ConvertDataType(tensor_type->dtype_);

    // Generate unique type names for this parameter
    std::string shape_type_name = base_name + "ShapeDim5";
    std::string stride_type_name = base_name + "StrideDim5";
    std::string global_type_name = base_name + "GlobalType";

    // Generate Shape type alias
    std::string shape_type = type_converter_.GenerateShapeType(shape_dims);
    std::string shape_alias = "using ";
    shape_alias += shape_type_name;
    shape_alias += " = ";
    shape_alias += shape_type;
    shape_alias += ";";
    emitter_.EmitLine(shape_alias);

    // Generate Stride type alias
    std::string stride_type = type_converter_.GenerateStrideType(shape_dims);
    std::string stride_alias = "using ";
    stride_alias += stride_type_name;
    stride_alias += " = ";
    stride_alias += stride_type;
    stride_alias += ";";
    emitter_.EmitLine(stride_alias);

    // Generate GlobalTensor type alias
    std::string global_type_alias = "using ";
    global_type_alias += global_type_name;
    global_type_alias += " = GlobalTensor<";
    global_type_alias += element_type;
    global_type_alias += ", ";
    global_type_alias += shape_type_name;
    global_type_alias += ", ";
    global_type_alias += stride_type_name;
    global_type_alias += ">;";
    emitter_.EmitLine(global_type_alias);

    // Generate GlobalTensor instance
    std::string global_instance = global_type_name;
    global_instance += " ";
    global_instance += global_name;
    global_instance += "(";
    global_instance += base_name;
    global_instance += ");";
    emitter_.EmitLine(global_instance);
  }

  emitter_.EmitLine("");

  // Collect all TileType variables from function body
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> tile_vars;
  if (func->body_) {
    tile_vars = CollectTileVariables(func->body_);
  }

  // Generate Tile type definitions and allocations
  if (!tile_vars.empty()) {
    emitter_.EmitLine("// Tile type definitions and allocations");

    for (const auto& [var, tile_type] : tile_vars) {
      // Just use sanitized name (will be registered later in AssignStmt)
      const std::string var_name = context_.SanitizeName(var);

      // Extract tile shape dimensions
      std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);

      // Get element type
      std::string element_type = type_converter_.ConvertDataType(tile_type->dtype_);

      // Determine tile dimensions (default to 1 if not specified)
      int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
      int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;

      // Generate Tile type alias
      std::string type_alias_name = var_name + "Type";
      std::string type_alias = "using ";
      type_alias += type_alias_name;
      type_alias += " = Tile<TileType::Vec, ";
      type_alias += element_type;
      type_alias += ", ";
      type_alias += std::to_string(rows);
      type_alias += ", ";
      type_alias += std::to_string(cols);
      type_alias += ", BLayout::RowMajor, -1, -1>;";
      emitter_.EmitLine(type_alias);

      // Generate Tile instance
      std::string tile_instance = type_alias_name;
      tile_instance += " ";
      tile_instance += var_name;
      tile_instance += "(";
      tile_instance += std::to_string(rows);
      tile_instance += ", ";
      tile_instance += std::to_string(cols);
      tile_instance += ");";
      emitter_.EmitLine(tile_instance);

      // Generate TASSIGN if MemRef is present
      if (tile_type->memref_.has_value()) {
        const auto memref_ptr = tile_type->memref_.value();
        int64_t addr = ExtractConstInt(memref_ptr->addr_);
        std::string tassign = "TASSIGN(";
        tassign += var_name;
        tassign += ", ";
        tassign += FormatAddressHex(addr);
        tassign += ");";
        emitter_.EmitLine(tassign);
      }
    }

    emitter_.EmitLine("");
  }

  emitter_.EmitLine("");
}

void CodeGenerator::GenerateBody(const ir::FunctionPtr& func) {
  emitter_.EmitLine("// Function body");
  if (func->body_) {
    VisitStmt(func->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
}

void CodeGenerator::VisitStmt_(const ir::AssignStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null AssignStmt";
  INTERNAL_CHECK(op->var_ != nullptr) << "Internal error: AssignStmt has null variable";
  INTERNAL_CHECK(op->value_ != nullptr) << "Internal error: AssignStmt has null value";

  // Sanitize and register the variable name
  std::string var_name = context_.SanitizeName(op->var_);
  context_.RegisterVar(op->var_, var_name);

  // Check if the value is a Call expression (block operation or sync operation)
  if (auto call = std::dynamic_pointer_cast<const ir::Call>(op->value_)) {
    // Convert kwargs vector to map for GetMapping
    // NOTE: For now, we pass empty attrs_map as ISA mapping doesn't currently need kwargs
    std::map<std::string, ir::ExprPtr> attrs_map;

    // Get ISA mapping for the operation
    auto mapping_opt = isa_mapper_.GetMapping(call->op_->name_, attrs_map);
    if (!mapping_opt.has_value()) {
      throw pypto::ValueError("No ISA mapping found for operation: " + call->op_->name_);
    }

    const auto& mapping = mapping_opt.value();

    // Helper lambda to compute offset expression
    // Formula: row_offset * stride + col_offset
    auto compute_offset = [this](const std::string& row, const std::string& col,
                                  const std::string& stride_expr) -> std::string {
      return row + " * " + stride_expr + " + " + col;
    };

    // Generate appropriate code based on operation type
    if (call->op_->name_ == "block.load") {
      // Pattern: compute offset, TASSIGN global tensor, then TLOAD
      // Args: tensor, row_offset, col_offset, height, width
      INTERNAL_CHECK(call->args_.size() == 5) << "block.load expects 5 arguments";

      // Get names
      std::string src_tensor_var = GetExprName(call->args_[0]);  // IR variable (parameter)
      std::string row_offset = GetExprName(call->args_[1]);
      std::string col_offset = GetExprName(call->args_[2]);

      // Get the tensor type to extract stride
      auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(call->args_[0]->GetType());
      INTERNAL_CHECK(tensor_type) << "First argument to block.load must be TensorType";

      // Extract stride from tensor shape
      // For 1D: stride = 1, For 2D+: stride = shape[-1]
      INTERNAL_CHECK(tensor_type->shape_.size() >= 1) << "Tensor must be at least 1D";
      std::string stride_expr;
      if (tensor_type->shape_.size() == 1) {
        stride_expr = "1";  // 1D tensor: no second dimension, stride is 1
      } else {
        // 2D+ tensor: stride is the width (last dimension)
        auto stride_ir_expr = tensor_type->shape_[tensor_type->shape_.size() - 1];
        stride_expr = GetExprName(stride_ir_expr);  // Can be ConstInt or Var
      }

      // Get raw pointer name (base name without "Global" suffix)
      std::string base_ptr = context_.SanitizeName(std::dynamic_pointer_cast<const ir::Var>(call->args_[0]));

      // Compute offset and emit TASSIGN to GlobalTensor
      std::string offset = compute_offset(row_offset, col_offset, stride_expr);
      emitter_.EmitLine("TASSIGN(" + src_tensor_var + ", " + base_ptr + " + " + offset + ");");

      // Emit TLOAD
      emitter_.EmitLine(mapping.isa_name + "(" + var_name + ", " + src_tensor_var + ");");

    } else if (call->op_->name_ == "block.store") {
      // Pattern: compute offset, TASSIGN global tensor, then TSTORE
      // Args: tile, row_offset, col_offset, height, width, output_tensor
      INTERNAL_CHECK(call->args_.size() == 6) << "block.store expects 6 arguments";

      std::string src_tile = GetExprName(call->args_[0]);
      std::string row_offset = GetExprName(call->args_[1]);
      std::string col_offset = GetExprName(call->args_[2]);
      std::string dst_tensor_var = GetExprName(call->args_[5]);

      // Get the tensor type to extract stride
      auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(call->args_[5]->GetType());
      INTERNAL_CHECK(tensor_type) << "Last argument to block.store must be TensorType";

      // Extract stride from tensor shape
      INTERNAL_CHECK(tensor_type->shape_.size() >= 1) << "Tensor must be at least 1D";
      std::string stride_expr;
      if (tensor_type->shape_.size() == 1) {
        stride_expr = "1";  // 1D tensor
      } else {
        auto stride_ir_expr = tensor_type->shape_[tensor_type->shape_.size() - 1];
        stride_expr = GetExprName(stride_ir_expr);  // Can be ConstInt or Var
      }

      // Get raw pointer name (base name without "Global" suffix)
      std::string base_ptr = context_.SanitizeName(std::dynamic_pointer_cast<const ir::Var>(call->args_[5]));

      // Compute offset and emit TASSIGN to GlobalTensor
      std::string offset = compute_offset(row_offset, col_offset, stride_expr);
      emitter_.EmitLine("TASSIGN(" + dst_tensor_var + ", " + base_ptr + " + " + offset + ");");

      // Emit TSTORE
      emitter_.EmitLine(mapping.isa_name + "(" + dst_tensor_var + ", " + src_tile + ");");

    } else {
      // Element-wise operations: TADD(dst, src0, src1) or TSQRT(dst, src) etc.
      std::ostringstream args_str;
      args_str << var_name;  // destination

      for (const auto& argExpr: call->args_) {
        args_str << ", " << GetExprName(argExpr);
      }

      emitter_.EmitLine(mapping.isa_name + "(" + args_str.str() + ");");
    }
  } else {
    // Non-call assignment (e.g., var = other_var)
    std::string value_name = GetExprName(op->value_);
    emitter_.EmitLine("auto " + var_name + " = " + value_name + ";");
  }
}

void CodeGenerator::VisitStmt_(const ir::EvalStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";
  INTERNAL_CHECK(op->expr_ != nullptr) << "Internal error: EvalStmt has null expression";

  // EvalStmt is used for expressions evaluated for side effects (no result assignment)
  // Currently used for sync operations (set_flag, wait_flag, barriers)
  if (auto call = std::dynamic_pointer_cast<const ir::Call>(op->expr_)) {
    // Convert kwargs vector to map for GetMapping
    std::map<std::string, ir::ExprPtr> attrs_map;

    // Get ISA mapping for the operation
    auto mapping_opt = isa_mapper_.GetMapping(call->op_->name_, attrs_map);
    if (!mapping_opt.has_value()) {
      throw pypto::ValueError("No ISA mapping found for operation: " + call->op_->name_);
    }

    const auto& mapping = mapping_opt.value();

    // Sync and barrier operations: emit function call with kwargs as arguments
    std::vector<std::string> args;
    for (const auto& [key, value] : call->kwargs_) {
      int arg_value = std::any_cast<int>(value);
      args.push_back(std::to_string(arg_value));
    }

    std::string args_str = args.empty() ? "" : args[0];
    for (size_t i = 1; i < args.size(); ++i) {
      args_str += ", " + args[i];
    }

    emitter_.EmitLine(mapping.isa_name + "(" + args_str + ");");
  } else {
    throw pypto::ValueError("EvalStmt with non-Call expression not yet supported");
  }
}

void CodeGenerator::VisitStmt_(const ir::ReturnStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ReturnStmt";
  // For void functions, we don't need to generate anything
  // The function will return implicitly at the closing brace
}

std::string CodeGenerator::GetExprName(const ir::ExprPtr& expr) {
  if (auto var = std::dynamic_pointer_cast<const ir::Var>(expr)) {
    return context_.GetVarName(var);
  }

  if (auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(expr)) {
    return std::to_string(const_int->value_);
  }

  if (auto const_float = std::dynamic_pointer_cast<const ir::ConstFloat>(expr)) {
    return std::to_string(const_float->value_);
  }

  if (auto const_bool = std::dynamic_pointer_cast<const ir::ConstBool>(expr)) {
    return const_bool->value_ ? "true" : "false";
  }

  // Handle binary expressions (a + b, a * b, etc.)
  if (auto binary = std::dynamic_pointer_cast<const ir::BinaryExpr>(expr)) {
    std::string left = GetExprName(binary->left_);
    std::string right = GetExprName(binary->right_);

    // Special handling for function-based binary operators
    if (std::dynamic_pointer_cast<const ir::Min>(expr)) {
      return "min(" + left + ", " + right + ")";
    }
    if (std::dynamic_pointer_cast<const ir::Max>(expr)) {
      return "max(" + left + ", " + right + ")";
    }
    if (std::dynamic_pointer_cast<const ir::Pow>(expr)) {
      return "pow(" + left + ", " + right + ")";
    }

    // Handle infix binary operators: +, -, *, /, etc.
    std::string op = GetBinaryOperator(expr);
    return "(" + left + " " + op + " " + right + ")";
  }

  // Handle unary expressions (-a, ~a, etc.)
  if (auto unary = std::dynamic_pointer_cast<const ir::UnaryExpr>(expr)) {
    std::string operand = GetExprName(unary->operand_);

    // Special handling for Cast - generate C++ cast
    if (auto cast = std::dynamic_pointer_cast<const ir::Cast>(expr)) {
      auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(cast->GetType());
      INTERNAL_CHECK(scalar_type) << "Cast expression must have ScalarType";
      std::string cpp_type = type_converter_.ConvertDataType(scalar_type->dtype_);
      return "((" + cpp_type + ")" + operand + ")";
    }

    // Special handling for Abs - generate function call
    if (std::dynamic_pointer_cast<const ir::Abs>(expr)) {
      return "abs(" + operand + ")";
    }

    // Handle prefix unary operators: -, !, ~
    std::string op = GetUnaryOperator(expr);
    return "(" + op + operand + ")";
  }

  // For other expressions, we would need to handle them appropriately
  throw pypto::ValueError("Unsupported expression type in codegen: " + expr->TypeName());
}

std::string CodeGenerator::GetBinaryOperator(const ir::ExprPtr& expr) {
  if (std::dynamic_pointer_cast<const ir::Add>(expr)) return "+";
  if (std::dynamic_pointer_cast<const ir::Sub>(expr)) return "-";
  if (std::dynamic_pointer_cast<const ir::Mul>(expr)) return "*";
  if (std::dynamic_pointer_cast<const ir::FloatDiv>(expr)) return "/";
  if (std::dynamic_pointer_cast<const ir::FloorDiv>(expr)) return "/";  // C++ integer division
  if (std::dynamic_pointer_cast<const ir::FloorMod>(expr)) return "%";
  if (std::dynamic_pointer_cast<const ir::Eq>(expr)) return "==";
  if (std::dynamic_pointer_cast<const ir::Ne>(expr)) return "!=";
  if (std::dynamic_pointer_cast<const ir::Lt>(expr)) return "<";
  if (std::dynamic_pointer_cast<const ir::Le>(expr)) return "<=";
  if (std::dynamic_pointer_cast<const ir::Gt>(expr)) return ">";
  if (std::dynamic_pointer_cast<const ir::Ge>(expr)) return ">=";
  if (std::dynamic_pointer_cast<const ir::And>(expr)) return "&&";
  if (std::dynamic_pointer_cast<const ir::Or>(expr)) return "||";
  if (std::dynamic_pointer_cast<const ir::BitAnd>(expr)) return "&";
  if (std::dynamic_pointer_cast<const ir::BitOr>(expr)) return "|";
  if (std::dynamic_pointer_cast<const ir::BitXor>(expr)) return "^";
  if (std::dynamic_pointer_cast<const ir::BitShiftLeft>(expr)) return "<<";
  if (std::dynamic_pointer_cast<const ir::BitShiftRight>(expr)) return ">>";
  throw pypto::ValueError("Unknown binary operator: " + expr->TypeName());
}

std::string CodeGenerator::GetUnaryOperator(const ir::ExprPtr& expr) {
  if (std::dynamic_pointer_cast<const ir::Neg>(expr)) return "-";
  if (std::dynamic_pointer_cast<const ir::Not>(expr)) return "!";
  if (std::dynamic_pointer_cast<const ir::BitNot>(expr)) return "~";
  throw pypto::ValueError("Unknown unary operator: " + expr->TypeName());
}

int64_t CodeGenerator::ExtractConstInt(const ir::ExprPtr& expr) {
  auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(expr);
  CHECK(const_int != nullptr) << "Expected constant integer expression";
  return const_int->value_;
}

namespace {

/**
 * @brief Helper visitor for collecting TileType variables from IR
 *
 * Traverses the IR tree and collects all variables with TileType.
 * Uses the visitor pattern for clean, extensible traversal.
 */
class TileCollector : public ir::IRVisitor {
 public:
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> tile_vars_;

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    // Check if the assigned variable has TileType
    auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(op->var_->GetType());
    if (tile_type) {
      tile_vars_.emplace_back(op->var_, tile_type);
    }
  }

};

}  // namespace

std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CodeGenerator::CollectTileVariables(
    const ir::StmtPtr& stmt) {
  if (!stmt) {
    return {};
  }

  TileCollector collector;
  collector.VisitStmt(stmt);
  return collector.tile_vars_;
}

std::vector<int64_t> CodeGenerator::ExtractShapeDimensions(
    const std::vector<ir::ExprPtr>& shape_exprs) {
  std::vector<int64_t> dims;
  dims.reserve(shape_exprs.size());
  for (const auto& expr : shape_exprs) {
    dims.push_back(ExtractConstInt(expr));
  }
  return dims;
}

std::string CodeGenerator::FormatAddressHex(int64_t addr) {
  std::ostringstream oss;
  oss << "0x" << std::hex << addr;
  return oss.str();
}

}  // namespace codegen

}  // namespace pypto
