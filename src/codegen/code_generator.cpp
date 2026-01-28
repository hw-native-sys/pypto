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
    std::ostringstream unpacking_line;
    unpacking_line << "__gm__ " << element_type << "* " << param_name
                   << " = reinterpret_cast<__gm__ " << element_type << "*>(args[" << i << "]);";
    emitter_.EmitLine(unpacking_line.str());
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

    // Generate GlobalTensor type and declaration with base pointer mapping
    GenerateGlobalTensorTypeDeclaration(global_name, tensor_type, base_name);
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

      // Generate Tile type and declaration (memref extracted automatically from tile_type)
      GenerateTileTypeDeclaration(var_name, tile_type);
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

  // Set context for expression visitor (dual-mode pattern)
  current_target_var_ = var_name;
  current_expr_value_ = "";

  // Dispatch to type-specific expression handler
  VisitExpr(op->value_);

  // If expression was non-Call (returned a value), emit assignment
  if (!current_expr_value_.empty()) {
    emitter_.EmitLine("auto " + var_name + " = " + current_expr_value_ + ";");
    current_expr_value_ = "";
  }

  // Clear context
  current_target_var_ = "";
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

void CodeGenerator::VisitStmt_(const ir::YieldStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null YieldStmt";

  if (op->value_.empty()) {
    return;  // No values to yield
  }

  // Visit each yielded expression and collect values
  std::vector<std::string> yielded_values;
  for (const auto& expr : op->value_) {
    VisitExpr(expr);
    yielded_values.push_back(current_expr_value_);
  }

  // Store in temporary buffer for ForStmt to pick up
  yield_buffer_ = yielded_values;
  current_expr_value_ = "";
}

void CodeGenerator::VisitStmt_(const ir::IfStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";
  INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: IfStmt has null condition";
  INTERNAL_CHECK(op->then_body_ != nullptr) << "Internal error: IfStmt has null then_body";

  // Declare and register return variables BEFORE the if statement
  for (const auto& return_var : op->return_vars_) {
    std::string return_var_name = context_.SanitizeName(return_var);
    context_.RegisterVar(return_var, return_var_name);

    // Generate appropriate type declaration based on variable type
    if (auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(return_var->GetType())) {
      // Tile variables are uninitialized at declaration (no memref_addr)
      GenerateTileTypeDeclaration(return_var_name, tile_type);
    } else if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(return_var->GetType())) {
      // GlobalTensor variables are uninitialized at declaration (no base_pointer)
      GenerateGlobalTensorTypeDeclaration(return_var_name, tensor_type);
    } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(return_var->GetType())) {
      // Generate scalar type declaration
      std::string cpp_type = type_converter_.ConvertDataType(scalar_type->dtype_);
      emitter_.EmitLine(cpp_type + " " + return_var_name + ";");
    } else {
      throw pypto::RuntimeError("Unsupported return_var type in IfStmt");
    }
  }

  if (!op->return_vars_.empty()) {
    emitter_.EmitLine("");  // Blank line for clarity
  }

  // Evaluate condition
  VisitExpr(op->condition_);
  std::string condition = current_expr_value_;
  current_expr_value_ = "";

  // Emit if statement
  emitter_.EmitLine("if (" + condition + ") {");
  emitter_.IncreaseIndent();

  // Visit then branch
  VisitStmt(op->then_body_);

  // Assign return variables from then branch yield
  if (!op->return_vars_.empty() && !yield_buffer_.empty()) {
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      std::string return_var_name = context_.SanitizeName(return_var);
      std::string yielded_value = yield_buffer_[i];

      emitter_.EmitLine(return_var_name + " = " + yielded_value + ";");

      // If the yielded value is a TensorType (GlobalTensor), inherit pointer mapping
      if (std::dynamic_pointer_cast<const ir::TensorType>(return_var->GetType())) {
        std::string yielded_ptr = context_.GetPointer(yielded_value);
        context_.RegisterPointer(return_var_name, yielded_ptr);
      }
    }
    yield_buffer_.clear();
  }

  emitter_.DecreaseIndent();

  // Emit else branch if present
  if (op->else_body_.has_value()) {
    emitter_.EmitLine("} else {");
    emitter_.IncreaseIndent();

    VisitStmt(op->else_body_.value());

    // Assign return variables from else branch yield
    if (!op->return_vars_.empty() && !yield_buffer_.empty()) {
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        const auto& return_var = op->return_vars_[i];
        std::string return_var_name = context_.SanitizeName(return_var);
        std::string yielded_value = yield_buffer_[i];

        emitter_.EmitLine(return_var_name + " = " + yielded_value + ";");

        // If the yielded value is a TensorType (GlobalTensor), inherit pointer mapping
        if (std::dynamic_pointer_cast<const ir::TensorType>(return_var->GetType())) {
          std::string yielded_ptr = context_.GetPointer(yielded_value);
          context_.RegisterPointer(return_var_name, yielded_ptr);
        }
      }
      yield_buffer_.clear();
    }

    emitter_.DecreaseIndent();
  }

  emitter_.EmitLine("}");
  emitter_.EmitLine("");
}

void CodeGenerator::VisitStmt_(const ir::ForStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";
  INTERNAL_CHECK(op->loop_var_ != nullptr) << "Internal error: ForStmt has null loop_var";
  INTERNAL_CHECK(op->start_ != nullptr) << "Internal error: ForStmt has null start";
  INTERNAL_CHECK(op->stop_ != nullptr) << "Internal error: ForStmt has null stop";
  INTERNAL_CHECK(op->step_ != nullptr) << "Internal error: ForStmt has null step";
  INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: ForStmt has null body";

  // Check consistency: iter_args and return_vars must have same size
  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "ForStmt iter_args size (" << op->iter_args_.size()
      << ") must equal return_vars size (" << op->return_vars_.size() << ")";

  // Register loop variable
  std::string loop_var_name = context_.SanitizeName(op->loop_var_);
  context_.RegisterVar(op->loop_var_, loop_var_name);

  // Register iteration arguments (loop-carried values)
  std::vector<std::string> iter_arg_names;
  if (!op->iter_args_.empty()) {
    emitter_.EmitLine("// Loop-carried values initialization");
    for (auto& iter_arg : op->iter_args_) {
      std::string iter_arg_name = context_.SanitizeName(iter_arg);
      context_.RegisterVar(iter_arg, iter_arg_name);
      iter_arg_names.push_back(iter_arg_name);

      // Initialize from IterArg's initValue
      VisitExpr(iter_arg->initValue_);
      std::string init_value = current_expr_value_;
      current_expr_value_ = "";

      // If initializing from a tensor variable, inherit its pointer mapping
      auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
      if (init_var) {
        std::string init_var_name = context_.GetVarName(init_var);
        std::string init_ptr = context_.GetPointer(init_var_name);
        
        context_.RegisterPointer(iter_arg_name, init_ptr);
      }

      iter_arg_name += " = ";
      iter_arg_name += init_value;
      iter_arg_name += ";";
      emitter_.EmitLine("auto " + iter_arg_name);
    }
    emitter_.EmitLine("");
  }

  // Evaluate loop range
  VisitExpr(op->start_);
  std::string start = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->stop_);
  std::string stop = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->step_);
  std::string step = current_expr_value_;
  current_expr_value_ = "";

  // Emit for loop
  emitter_.EmitLine("for (int64_t " + loop_var_name + " = " + start + "; " +
                    loop_var_name + " < " + stop + "; " +
                    loop_var_name + " += " + step + ") {");
  emitter_.IncreaseIndent();

  // Visit loop body
  // Note: Do NOT clear yield_buffer before visiting body - it should persist
  // across the body visit but be cleared AFTER in case of multiple loop executions
  yield_buffer_.clear();
  VisitStmt(op->body_);

  // If iter_args exist and yield was called, assign yielded values
  if (!op->iter_args_.empty() && !yield_buffer_.empty()) {
    CHECK(yield_buffer_.size() == iter_arg_names.size())
        << "Yielded " << yield_buffer_.size() << " values but expected "
        << iter_arg_names.size();

    for (size_t i = 0; i < iter_arg_names.size(); ++i) {
      emitter_.EmitLine(iter_arg_names[i] + " = " + yield_buffer_[i] + ";");
    }
  }
  yield_buffer_.clear();

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
  emitter_.EmitLine("");

  // Register return variables with same names as iter_args
  // (return_vars represent the final values of iter_args after loop completion)
  if (!op->return_vars_.empty()) {
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      // Register return_var with the iter_arg's name
      // They represent the same value, so no assignment needed
      if (i < iter_arg_names.size()) {
        context_.RegisterVar(return_var, iter_arg_names[i]);
      } else {
        // No corresponding iter_arg (shouldn't happen if CHECK above passes)
        throw pypto::RuntimeError("ForStmt return_var has no corresponding iter_arg");
      }
    }
  }
}

// ========================================================================
// Expression Visitor Methods - Dual-Mode Pattern
// ========================================================================
// - Statement-Emitting Mode (Call): Uses current_target_var_, emits instructions
// - Value-Returning Mode (others): Sets current_expr_value_ with inline C++ code

// ---- Leaf Nodes ----

void CodeGenerator::VisitExpr_(const ir::VarPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Var";
  current_expr_value_ = context_.GetVarName(op);
}

void CodeGenerator::VisitExpr_(const ir::IterArgPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IterArg";
  // IterArg inherits from Var, treated same way
  current_expr_value_ = context_.GetVarName(std::dynamic_pointer_cast<const ir::Var>(op));
}

void CodeGenerator::VisitExpr_(const ir::ConstIntPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstInt";
  current_expr_value_ = std::to_string(op->value_);
}

void CodeGenerator::VisitExpr_(const ir::ConstFloatPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstFloat";
  current_expr_value_ = std::to_string(op->value_);
}

void CodeGenerator::VisitExpr_(const ir::ConstBoolPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstBool";
  current_expr_value_ = op->value_ ? "true" : "false";
}

void CodeGenerator::VisitExpr_(const ir::TupleGetItemExprPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null TupleGetItemExpr";
  VisitExpr(op->tuple_);
  std::string tuple_name = current_expr_value_;
  current_expr_value_ = tuple_name + "[" + std::to_string(op->index_) + "]";
}

void CodeGenerator::VisitExpr_(const ir::CallPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Call";
  INTERNAL_CHECK(!current_target_var_.empty()) << "Internal error: Call without assignment target";

  // Use existing call handling logic (statement-emitting mode)
  std::string var_name = current_target_var_;

  auto mapping_opt = isa_mapper_.GetMapping(op->op_->name_, {});
  if (!mapping_opt.has_value()) {
    throw pypto::ValueError("No ISA mapping found for operation: " + op->op_->name_);
  }
  const auto& mapping = mapping_opt.value();

  // block.load and block.store require special handling
  if (op->op_->name_ == "block.load") {
    // block.load(tensor, row_offset, col_offset, height, width)
    CHECK(op->args_.size() == 5) << "block.load requires 5 arguments: tensor, row_offset, col_offset, height, "
                                      "width";

    // Get source tensor variable (arg 0)
    auto src_tensor_expr = op->args_[0];
    auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(src_tensor_expr);
    CHECK(src_tensor_var_ptr != nullptr) << "block.load source tensor must be a Var";

    std::string src_tensor_var = context_.GetVarName(src_tensor_var_ptr);

    // Get offsets
    VisitExpr(op->args_[1]);
    std::string row_offset = current_expr_value_;
    VisitExpr(op->args_[2]);
    std::string col_offset = current_expr_value_;

    // Compute offset using row_offset * stride + col_offset
    auto src_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(src_tensor_var_ptr->GetType());
    CHECK(src_tensor_type != nullptr) << "block.load source must be TensorType";
    INTERNAL_CHECK(src_tensor_type->shape_.size() >= 1) << "Tensor must be at least 1D";

    // Calculate stride (width of last dimension)
    std::string stride_expr;
    if (src_tensor_type->shape_.size() == 1) {
      stride_expr = "1";
    } else {
      auto stride_ir_expr = src_tensor_type->shape_[src_tensor_type->shape_.size() - 1];
      VisitExpr(stride_ir_expr);
      stride_expr = current_expr_value_;
    }

    std::string offset = row_offset + " * " + stride_expr + " + " + col_offset;

    // Emit TASSIGN to set the start address, then TLOAD
    // Use GetPointer to get the raw pointer name for address computation
    std::string raw_ptr = context_.GetPointer(src_tensor_var);
    emitter_.EmitLine("TASSIGN(" + src_tensor_var + ", " + raw_ptr + " + " + offset + ");");
    emitter_.EmitLine(mapping.isa_name + "(" + var_name + ", " + src_tensor_var + ");");
  } else if (op->op_->name_ == "block.store") {
    // block.store(tile, row_offset, col_offset, height, width, output_tensor)
    CHECK(op->args_.size() == 6) << "block.store requires 6 arguments: tile, row_offset, col_offset, height, "
                                      "width, output_tensor";

    // Get source tile (arg 0)
    VisitExpr(op->args_[0]);
    std::string src_tile = current_expr_value_;

    // Get offsets
    VisitExpr(op->args_[1]);
    std::string row_offset = current_expr_value_;
    VisitExpr(op->args_[2]);
    std::string col_offset = current_expr_value_;

    // Get output tensor (arg 5)
    auto dst_tensor_expr = op->args_[5];
    auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(dst_tensor_expr);
    CHECK(dst_tensor_var_ptr != nullptr) << "block.store destination tensor must be a Var";

    std::string dst_tensor_var = context_.GetVarName(dst_tensor_var_ptr);

    // Compute offset using row_offset * stride + col_offset
    auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
    CHECK(dst_tensor_type != nullptr) << "block.store destination must be TensorType";
    INTERNAL_CHECK(dst_tensor_type->shape_.size() >= 1) << "Tensor must be at least 1D";

    // Calculate stride
    std::string stride_expr;
    if (dst_tensor_type->shape_.size() == 1) {
      stride_expr = "1";
    } else {
      auto stride_ir_expr = dst_tensor_type->shape_[dst_tensor_type->shape_.size() - 1];
      VisitExpr(stride_ir_expr);
      stride_expr = current_expr_value_;
    }

    std::string offset = row_offset + " * " + stride_expr + " + " + col_offset;

    // Emit TASSIGN to set the start address, then TSTORE
    // Use GetPointer to get the raw pointer name for address computation
    std::string raw_ptr = context_.GetPointer(dst_tensor_var);
    emitter_.EmitLine("TASSIGN(" + dst_tensor_var + ", " + raw_ptr + " + " + offset + ");");
    emitter_.EmitLine(mapping.isa_name + "(" + dst_tensor_var + ", " + src_tile + ");");

    // Register pointer and generate assign for ssa output
    context_.RegisterPointer(var_name, dst_tensor_var);
    emitter_.EmitLine("auto " + var_name + " = " + dst_tensor_var + ";");
    
  } else {
    // Element-wise operations: TADD(dst, src0, src1) or TSQRT(dst, src) etc.
    std::ostringstream args_str;
    args_str << var_name;  // destination

    for (const auto& argExpr : op->args_) {
      VisitExpr(argExpr);
      args_str << ", " << current_expr_value_;
    }

    emitter_.EmitLine(mapping.isa_name + "(" + args_str.str() + ");");
  }

  // Don't set current_expr_value_ - indicates statement-emitting mode
  current_expr_value_ = "";
}

// ---- Binary Operators ----

#define IMPLEMENT_BINARY_OP(OpType, OpName, CppOp)                       \
  void CodeGenerator::VisitExpr_(const ir::OpType##Ptr& op) {            \
    INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName);  \
    VisitExpr(op->left_);                                                 \
    std::string left = current_expr_value_;                               \
    VisitExpr(op->right_);                                                \
    std::string right = current_expr_value_;                              \
    current_expr_value_ = "(" + left + " " + (CppOp) + " " + right + ")";  \
  }

// Arithmetic operators
IMPLEMENT_BINARY_OP(Add, "Add", "+")
IMPLEMENT_BINARY_OP(Sub, "Sub", "-")
IMPLEMENT_BINARY_OP(Mul, "Mul", "*")
IMPLEMENT_BINARY_OP(FloorDiv, "FloorDiv", "/")
IMPLEMENT_BINARY_OP(FloorMod, "FloorMod", "%")
IMPLEMENT_BINARY_OP(FloatDiv, "FloatDiv", "/")

// Comparison operators
IMPLEMENT_BINARY_OP(Eq, "Eq", "==")
IMPLEMENT_BINARY_OP(Ne, "Ne", "!=")
IMPLEMENT_BINARY_OP(Lt, "Lt", "<")
IMPLEMENT_BINARY_OP(Le, "Le", "<=")
IMPLEMENT_BINARY_OP(Gt, "Gt", ">")
IMPLEMENT_BINARY_OP(Ge, "Ge", ">=")

// Logical operators
IMPLEMENT_BINARY_OP(And, "And", "&&")
IMPLEMENT_BINARY_OP(Or, "Or", "||")
IMPLEMENT_BINARY_OP(Xor, "Xor", "^")

// Bitwise operators
IMPLEMENT_BINARY_OP(BitAnd, "BitAnd", "&")
IMPLEMENT_BINARY_OP(BitOr, "BitOr", "|")
IMPLEMENT_BINARY_OP(BitXor, "BitXor", "^")
IMPLEMENT_BINARY_OP(BitShiftLeft, "BitShiftLeft", "<<")
IMPLEMENT_BINARY_OP(BitShiftRight, "BitShiftRight", ">>")

#undef IMPLEMENT_BINARY_OP

// Special binary operators (function calls)
void CodeGenerator::VisitExpr_(const ir::MinPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Min";
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = "min(" + left + ", " + right + ")";
}

void CodeGenerator::VisitExpr_(const ir::MaxPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Max";
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = "max(" + left + ", " + right + ")";
}

void CodeGenerator::VisitExpr_(const ir::PowPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Pow";
  VisitExpr(op->left_);
  std::string left = current_expr_value_;
  VisitExpr(op->right_);
  std::string right = current_expr_value_;
  current_expr_value_ = "pow(" + left + ", " + right + ")";
}

// ---- Unary Operators ----

#define IMPLEMENT_UNARY_OP(OpType, OpName, CppOp)                        \
  void CodeGenerator::VisitExpr_(const ir::OpType##Ptr& op) {            \
    INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName);     \
    VisitExpr(op->operand_);                                              \
    current_expr_value_ = std::string("(") + (CppOp) + current_expr_value_ + ")";  \
  }

IMPLEMENT_UNARY_OP(Neg, "Neg", "-")
IMPLEMENT_UNARY_OP(Not, "Not", "!")
IMPLEMENT_UNARY_OP(BitNot, "BitNot", "~")

#undef IMPLEMENT_UNARY_OP

// Special unary operators
void CodeGenerator::VisitExpr_(const ir::AbsPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Abs";
  VisitExpr(op->operand_);
  std::string operand = current_expr_value_;
  current_expr_value_ = "abs(" + operand + ")";
}

void CodeGenerator::VisitExpr_(const ir::CastPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Cast";
  VisitExpr(op->operand_);
  std::string operand = current_expr_value_;

  auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
  CHECK(scalar_type != nullptr) << "Cast target must be ScalarType";

  std::string cpp_type = type_converter_.ConvertDataType(scalar_type->dtype_);
  current_expr_value_ = "((" + cpp_type + ")" + operand + ")";
}

// ========================================================================
// End of Expression Visitor Methods
// ========================================================================


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

void CodeGenerator::GenerateTileTypeDeclaration(const std::string& var_name,
                                                   const ir::TileTypePtr& tile_type) {
  INTERNAL_CHECK(!var_name.empty()) << "Internal error: var_name cannot be empty";
  INTERNAL_CHECK(tile_type != nullptr) << "Internal error: tile_type is null";

  // Extract tile shape dimensions
  std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);

  // Get element type
  std::string element_type = type_converter_.ConvertDataType(tile_type->dtype_);

  // Determine tile dimensions (default to 1 if not specified)
  int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
  int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;

  // Generate Tile type alias
  std::string type_alias_name = var_name + "Type";
  std::ostringstream type_alias;
  type_alias << "using " << type_alias_name << " = Tile<TileType::Vec, "
             << element_type << ", " << rows << ", " << cols
             << ", BLayout::RowMajor, -1, -1>;";
  emitter_.EmitLine(type_alias.str());

  // Generate Tile instance
  std::ostringstream tile_instance;
  tile_instance << type_alias_name << " " << var_name << "(" << rows << ", " << cols << ");";
  emitter_.EmitLine(tile_instance.str());

  // Generate TASSIGN if tile_type has memref
  if (tile_type->memref_.has_value()) {
    const auto memref_ptr = tile_type->memref_.value();
    int64_t addr = ExtractConstInt(memref_ptr->addr_);
    std::ostringstream tassign;
    tassign << "TASSIGN(" << var_name << ", " << FormatAddressHex(addr) << ");";
    emitter_.EmitLine(tassign.str());
  }
}

void CodeGenerator::GenerateGlobalTensorTypeDeclaration(const std::string& var_name,
                                                           const ir::TensorTypePtr& tensor_type,
                                                           const std::optional<std::string>& base_pointer) {
  INTERNAL_CHECK(!var_name.empty()) << "Internal error: var_name cannot be empty";
  INTERNAL_CHECK(tensor_type != nullptr) << "Internal error: tensor_type is null";

  // Extract shape dimensions
  std::vector<int64_t> shape_dims;
  shape_dims.reserve(tensor_type->shape_.size());
  for (const auto& dim_expr : tensor_type->shape_) {
    shape_dims.push_back(ExtractConstInt(dim_expr));
  }

  // Get element type
  std::string element_type = type_converter_.ConvertDataType(tensor_type->dtype_);

  // Generate unique type names for this variable
  std::string shape_type_name = var_name + "ShapeDim5";
  std::string stride_type_name = var_name + "StrideDim5";
  std::string global_type_name = var_name + "Type";

  // Generate Shape type alias
  std::string shape_type = type_converter_.GenerateShapeType(shape_dims);
  std::ostringstream shape_alias;
  shape_alias << "using " << shape_type_name << " = " << shape_type << ";";
  emitter_.EmitLine(shape_alias.str());

  // Generate Stride type alias
  std::string stride_type = type_converter_.GenerateStrideType(shape_dims);
  std::ostringstream stride_alias;
  stride_alias << "using " << stride_type_name << " = " << stride_type << ";";
  emitter_.EmitLine(stride_alias.str());

  // Generate GlobalTensor type alias
  std::ostringstream global_type_alias;
  global_type_alias << "using " << global_type_name << " = GlobalTensor<"
                    << element_type << ", " << shape_type_name << ", "
                    << stride_type_name << ">;";
  emitter_.EmitLine(global_type_alias.str());

  // Generate GlobalTensor instance
  std::ostringstream global_instance;
  global_instance << global_type_name << " " << var_name << "(";
  if (base_pointer.has_value()) {
    global_instance << base_pointer.value();
  }
  global_instance << ");";
  emitter_.EmitLine(global_instance.str());

  // Register pointer mapping if base_pointer provided
  if (base_pointer.has_value()) {
    context_.RegisterPointer(var_name, base_pointer.value());
  }
}

}  // namespace codegen

}  // namespace pypto
