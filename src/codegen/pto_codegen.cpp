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

#include "pypto/codegen/pto_codegen.h"

#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto {
namespace ir {

namespace {

// Internal visitor to collect tile and scalar declarations in one pass
// After AddAllocPass, we only generate declarations for block.alloc operations (MemRef allocations)
// Regular TileType variables are mapped to their corresponding MemRef and don't need separate declarations
class DeclarationCollector : public IRVisitor {
 public:
  DeclarationCollector(std::map<std::string, TileInfo>& tile_decls,
                       std::vector<std::pair<std::string, DataType>>& scalar_decls,
                       std::map<std::string, std::string>& var_to_physical_tile,
                       std::map<uint64_t, std::string>& memref_to_var)
      : tile_decls_(tile_decls),
        scalar_decls_(scalar_decls),
        var_to_physical_tile_(var_to_physical_tile),
        memref_to_var_(memref_to_var) {}

  // After visiting all statements, finalize the declarations
  void Finalize() {
    // Create tile declarations only for block.alloc operations (MemRef allocations)
    for (const auto& [memref_id, tile_info] : memref_id_to_tile_info_) {
      if (memref_id_to_memref_var_.count(memref_id)) {
        std::string memref_var_name = memref_id_to_memref_var_[memref_id];
        TileInfo info = tile_info;
        info.name = memref_var_name;
        tile_decls_[memref_var_name] = info;

        // Update var_to_physical_tile mapping:
        // All tile variables using this MemRef should map to the MemRef variable name
        var_to_physical_tile_[memref_var_name] = memref_var_name;

        // Map all tile variables that use this MemRef ID to the MemRef variable name
        for (const auto& [tile_var, tile_memref_id] : tile_var_to_memref_id_) {
          if (tile_memref_id == memref_id) {
            var_to_physical_tile_[tile_var] = memref_var_name;
          }
        }
      }
    }
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    // Check if this is a block.alloc operation
    if (auto call = As<Call>(op->value_)) {
      if (call->op_->name_ == "block.alloc") {
        // Extract MemRef ID from args[3]
        if (call->args_.size() >= 4) {
          if (auto id_expr = As<ConstInt>(call->args_[3])) {
            uint64_t memref_id = static_cast<uint64_t>(id_expr->value_);

            // Record the MemRef variable name for this ID
            memref_id_to_memref_var_[memref_id] = op->var_->name_;
          }
        }
      }
    }

    // Build MemRef ID -> TileInfo mapping from TileType variables
    if (auto tile_type = As<TileType>(op->var_->GetType())) {
      if (tile_type->memref_.has_value()) {
        const auto& memref = tile_type->memref_.value();
        uint64_t memref_id = memref->id_;

        // Store TileInfo for this MemRef ID (if not already stored)
        if (!memref_id_to_tile_info_.count(memref_id)) {
          TileInfo info;
          // Extract shape and dtype from TileType
          if (tile_type->shape_.size() >= 2) {
            if (auto rows_const = As<ConstInt>(tile_type->shape_[0])) {
              info.rows = rows_const->value_;
            } else {
              info.rows = 8;  // Default fallback
            }
            if (auto cols_const = As<ConstInt>(tile_type->shape_[1])) {
              info.cols = cols_const->value_;
            } else {
              info.cols = 8;  // Default fallback
            }
          } else {
            info.rows = 8;
            info.cols = 8;
          }
          info.dtype = tile_type->dtype_;
          memref_id_to_tile_info_[memref_id] = info;
        }

        // Record the mapping from tile variable to MemRef ID
        // This will be used in Finalize() to map tile vars to MemRef var names
        tile_var_to_memref_id_[op->var_->name_] = memref_id;
      } else {
        // No MemRef - this shouldn't happen after AddAllocPass, but handle it
        TileInfo info;
        info.name = op->var_->name_;
        if (tile_type->shape_.size() >= 2) {
          if (auto rows_const = As<ConstInt>(tile_type->shape_[0])) {
            info.rows = rows_const->value_;
          } else {
            info.rows = 8;
          }
          if (auto cols_const = As<ConstInt>(tile_type->shape_[1])) {
            info.cols = cols_const->value_;
          } else {
            info.cols = 8;
          }
        } else {
          info.rows = 8;
          info.cols = 8;
        }
        info.dtype = tile_type->dtype_;
        tile_decls_[info.name] = info;
        var_to_physical_tile_[op->var_->name_] = op->var_->name_;
      }
    }

    // Also check TensorType for MemRef mapping (for function params and loop results)
    if (auto tensor_type = As<TensorType>(op->var_->GetType())) {
      if (tensor_type->memref_.has_value()) {
        const auto& memref = tensor_type->memref_.value();
        uint64_t memref_id = memref->id_;

        // Only record the first variable with this MemRef ID in memref_to_var_
        // This ensures we always map back to the original variable (usually function parameter)
        if (!memref_to_var_.count(memref_id)) {
          memref_to_var_[memref_id] = op->var_->name_;
        }
      }
    }

    // Check if this variable has ScalarType
    if (auto scalar_type = As<ScalarType>(op->var_->GetType())) {
      // Check if this scalar is already declared (avoid duplicates)
      bool already_declared = false;
      for (const auto& [name, dtype] : scalar_decls_) {
        if (name == op->var_->name_) {
          already_declared = true;
          break;
        }
      }
      if (!already_declared) {
        scalar_decls_.push_back({op->var_->name_, scalar_type->dtype_});
      }
    }

    // Continue traversal of child nodes
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    // EvalStmt contains an expression that may need to be visited
    // But for declaration collection, we don't need to process it
    // Just provide empty implementation to satisfy abstract class requirement
  }

 private:
  std::map<std::string, TileInfo>& tile_decls_;
  std::vector<std::pair<std::string, DataType>>& scalar_decls_;
  std::map<std::string, std::string>& var_to_physical_tile_;
  std::map<uint64_t, std::string>& memref_to_var_;       // MemRef ID -> original var name (for tensors)
  std::map<uint64_t, TileInfo> memref_id_to_tile_info_;  // MemRef ID -> TileInfo (from TileType vars)
  std::map<uint64_t, std::string>
      memref_id_to_memref_var_;                            // MemRef ID -> MemRef var name (from block.alloc)
  std::map<std::string, uint64_t> tile_var_to_memref_id_;  // Tile var name -> MemRef ID (for mapping)
};

}  // namespace

std::string PTOCodegen::Generate(const ProgramPtr& program) {
  // Clear state
  code_lines_.clear();
  tile_decls_.clear();
  var_to_physical_tile_.clear();
  memref_to_var_.clear();
  indent_level_ = 0;
  current_function_name_.clear();

  // Generate code for each function
  std::vector<std::string> function_codes;
  for (const auto& [gvar, func] : program->functions_) {
    std::string func_code = GenerateFunction(func);
    function_codes.push_back(func_code);
  }

  // Combine all function codes
  std::ostringstream oss;
  oss << "// PTO Program: " << program->name_ << "\n";
  oss << "// Generated by PyPTO IR Compiler\n\n";

  for (const auto& func_code : function_codes) {
    oss << func_code << "\n";
  }

  return oss.str();
}

std::string PTOCodegen::GenerateFunction(const FunctionPtr& func) {
  // Reset state for this function
  code_lines_.clear();
  tile_decls_.clear();
  scalar_decls_.clear();
  var_to_physical_tile_.clear();  // Clear variable mapping
  memref_to_var_.clear();         // Clear MemRef mapping
  indent_level_ = 0;
  current_function_name_ = func->name_;

  // First, collect MemRef information from function parameters
  // This allows us to map IterArg types to their original parameter names
  for (const auto& param : func->params_) {
    if (auto tensor_type = As<TensorType>(param->GetType())) {
      if (tensor_type->memref_.has_value()) {
        const auto& memref = tensor_type->memref_.value();
        memref_to_var_[memref->id_] = param->name_;
      }
    }
  }

  // Collect tile and scalar declarations from function body in one pass
  DeclarationCollector collector(tile_decls_, scalar_decls_, var_to_physical_tile_, memref_to_var_);
  collector.VisitStmt(func->body_);
  collector.Finalize();  // Finalize declarations after visiting all statements

  // Generate function signature: func @name(%param: type, ...) {
  std::ostringstream params_str;
  bool first_param = true;
  for (const auto& param : func->params_) {
    if (!first_param) {
      params_str << ", ";
    }
    first_param = false;

    // Generate parameter: %name: type
    params_str << "%" << param->name_ << ": " << TypeToPTOType(param->GetType());
  }

  EmitLine("func @" + func->name_ + "(" + params_str.str() + ") {");
  indent_level_++;

  // Generate tile declarations
  if (!tile_decls_.empty()) {
    EmitLine("// Tile Declarations");
    for (const auto& [name, info] : tile_decls_) {
      std::string dtype_str = DataTypeToPTOType(info.dtype);
      EmitLine("%" + name + " = alloc_tile : !pto.tile<" + std::to_string(info.rows) + "x" +
               std::to_string(info.cols) + "x" + dtype_str + ">");
    }
    EmitLine("");
  }

  // Generate scalar declarations
  if (!scalar_decls_.empty()) {
    EmitLine("// Scalar Declarations");
    for (const auto& [name, dtype] : scalar_decls_) {
      std::string dtype_str = DataTypeToPTOType(dtype);
      EmitLine("%" + name + " = alloc_scalar : " + dtype_str);
    }
    EmitLine("");
  }

  // Generate function body by visiting statements
  if (!tile_decls_.empty() || !scalar_decls_.empty()) {
    EmitLine("// Instructions");
  }
  VisitStmt(func->body_);

  // Generate return statement
  EmitLine("");
  EmitLine("return");
  indent_level_--;
  EmitLine("}");

  // Combine all lines into a single string
  std::ostringstream oss;
  for (const auto& line : code_lines_) {
    oss << line << "\n";
  }
  return oss.str();
}

void PTOCodegen::EmitLine(const std::string& line) {
  std::string indented_line;
  for (int i = 0; i < indent_level_; ++i) {
    indented_line += "  ";  // 2 spaces per indent level (PTO style)
  }
  indented_line += line;
  code_lines_.push_back(indented_line);
}

std::string PTOCodegen::OpToPTOInstruction(const CallPtr& op, const std::string& result_var) {
  std::string op_name = op->op_->name_;

  // Extract operation name (e.g., "block.mul" -> "mul")
  std::string method_name;
  size_t dot_pos = op_name.find_last_of('.');
  if (dot_pos != std::string::npos) {
    method_name = op_name.substr(dot_pos + 1);
  } else {
    method_name = op_name;
  }

  // Map to PTO instruction names
  std::string pto_instr;
  if (method_name == "mul") {
    pto_instr = "tmul";
  } else if (method_name == "add") {
    pto_instr = "tadd";
  } else if (method_name == "sub") {
    pto_instr = "tsub";
  } else if (method_name == "div") {
    pto_instr = "tdiv";
  } else if (method_name == "muls") {
    pto_instr = "tmuls";
  } else if (method_name == "adds") {
    pto_instr = "tadds";
  } else if (method_name == "subs") {
    pto_instr = "tsubs";
  } else if (method_name == "divs") {
    pto_instr = "tdivs";
  } else if (method_name == "load") {
    pto_instr = "tload";
  } else if (method_name == "store") {
    pto_instr = "tstore";
  } else {
    // Default: prefix with 't' for tile operations
    pto_instr = "t" + method_name;
  }

  // Build argument list with % prefix for variables
  std::vector<std::string> args;
  for (const auto& arg : op->args_) {
    std::string arg_str = ExtractVarName(arg);
    // Add % prefix only for variables, not for constants
    if (auto var = As<Var>(arg)) {
      // First try to resolve via MemRef ID (for variables sharing MemRef with function parameters)
      std::string resolved_name = arg_str;
      if (auto tensor_type = As<TensorType>(var->GetType())) {
        if (tensor_type->memref_.has_value()) {
          const auto& memref = tensor_type->memref_.value();
          uint64_t memref_id = memref->id_;

          // Look up the original variable name from MemRef ID
          if (memref_to_var_.count(memref_id)) {
            resolved_name = memref_to_var_[memref_id];
          }
        }
      }

      // Then resolve to physical tile name for tile variables
      std::string physical_name = ResolvePhysicalTile(resolved_name);
      args.push_back("%" + physical_name);
    } else if (auto iter_arg = As<IterArg>(arg)) {
      // IterArg is a subclass of Var, but has its own kind
      // For IterArg, we need to resolve to the original tensor parameter using MemRef ID
      std::string original_name = arg_str;

      // Try to find the original variable name via MemRef ID
      if (auto tensor_type = As<TensorType>(iter_arg->GetType())) {
        if (tensor_type->memref_.has_value()) {
          const auto& memref = tensor_type->memref_.value();
          uint64_t memref_id = memref->id_;

          // Look up the original variable name from MemRef ID
          if (memref_to_var_.count(memref_id)) {
            original_name = memref_to_var_[memref_id];
          }
        }
      }

      std::string physical_name = ResolvePhysicalTile(original_name);
      args.push_back("%" + physical_name);
    } else if (auto const_int = As<ConstInt>(arg)) {
      args.push_back(arg_str);  // No % prefix for constants
    } else if (auto const_float = As<ConstFloat>(arg)) {
      args.push_back(arg_str);  // No % prefix for constants
    } else {
      args.push_back("%" + arg_str);  // Default: add % prefix
    }
  }

  // Get result type for type annotation
  std::string type_annotation = "";
  if (op->GetType()) {
    type_annotation = " : " + TypeToPTOType(op->GetType());
  }

  // Special handling for different instruction types
  std::ostringstream oss;

  if (pto_instr == "tstore") {
    // Store has no result assignment: tstore %tile, %output[row_offset, col_offset]
    // Args: [tile, row_offset, col_offset, height, width, output_tensor]
    if (args.size() >= 6) {
      oss << pto_instr << " " << args[0] << ", " << args[5];
      oss << "[" << args[1] << ", " << args[2] << "]";
    }
  } else if (pto_instr == "tload") {
    // Load: %dst = tload %src[idx, 0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
    oss << "%" << result_var << " = " << pto_instr;

    std::string src_memref_type;
    std::string index_type = "index";

    // Get source memref type from the first argument
    if (!args.empty() && !op->args_.empty()) {
      if (auto var = As<Var>(op->args_[0])) {
        src_memref_type = TypeToPTOType(var->GetType());
      }
    }

    if (!args.empty()) {
      oss << " " << args[0];
      if (args.size() > 1) {
        oss << "[" << args[1] << ", 0]";
      }
    }

    // Generate full type annotation: (!pto.memref<...>, index, index) -> !pto.tile<...>
    if (!src_memref_type.empty()) {
      oss << " : (" << src_memref_type << ", " << index_type << ", " << index_type << ") -> ";
    } else {
      oss << " : (...) -> ";
    }
    oss << TypeToPTOType(op->GetType());
  } else if (method_name.back() == 's' && (method_name == "muls" || method_name == "adds" ||
                                           method_name == "subs" || method_name == "divs")) {
    // Scalar operations: %result = tmuls %src, %scalar : !pto.tile<...>, scalar_type
    oss << "%" << result_var << " = " << pto_instr;
    if (!args.empty()) {
      oss << " " << args[0];
      // For scalar ops, second arg is the scalar constant
      if (args.size() > 1) {
        // Extract the actual constant value (no % prefix for constants)
        if (auto const_float = As<ConstFloat>(op->args_[1])) {
          oss << ", " << FormatFloat(const_float->value_);
        } else if (auto const_int = As<ConstInt>(op->args_[1])) {
          oss << ", " << const_int->value_;
        } else {
          oss << ", " << args[1];
        }
      }
    }
    oss << type_annotation << ", " << DataTypeToPTOType(DataType::FP32);
  } else {
    // Binary tile operations: %result = tmul %a, %b : !pto.tile<...>
    oss << "%" << result_var << " = " << pto_instr;
    for (size_t i = 0; i < args.size(); ++i) {
      if (i == 0) {
        oss << " ";
      } else {
        oss << ", ";
      }
      oss << args[i];
    }
    oss << type_annotation;
  }

  return oss.str();
}

std::string PTOCodegen::ExtractVarName(const ExprPtr& expr) {
  if (auto var = As<Var>(expr)) {
    return var->name_;
  } else if (auto iter_arg = As<IterArg>(expr)) {
    // IterArg is a subclass of Var with its own kind, handle it separately
    return iter_arg->name_;
  } else if (auto const_int = As<ConstInt>(expr)) {
    // Return string representation of integer constant
    return std::to_string(const_int->value_);
  } else if (auto const_float = As<ConstFloat>(expr)) {
    // Return string representation of float constant
    return FormatFloat(const_float->value_);
  } else {
    // For other expressions, try to generate a placeholder
    return "expr";
  }
}

std::string PTOCodegen::ResolvePhysicalTile(const std::string& var_name) {
  // Check if this variable has a mapping to a physical tile
  if (var_to_physical_tile_.count(var_name)) {
    return var_to_physical_tile_[var_name];
  }
  // No mapping found, return the original name
  return var_name;
}

std::string PTOCodegen::DataTypeToPTOType(DataType dtype) {
  // Map PyPTO DataType to PTO type strings
  if (dtype == DataType::FP32) {
    return "f32";
  } else if (dtype == DataType::FP16) {
    return "f16";
  } else if (dtype == DataType::BF16) {
    return "bf16";
  } else if (dtype == DataType::INT32) {
    return "i32";
  } else if (dtype == DataType::INT64) {
    return "i64";
  } else if (dtype == DataType::UINT8) {
    return "u8";
  } else if (dtype == DataType::INT8) {
    return "i8";
  } else if (dtype == DataType::BOOL) {
    return "u1";
  } else {
    return "f32";  // Default fallback
  }
}

std::string PTOCodegen::TypeToPTOType(const TypePtr& type) {
  if (auto tile_type = As<TileType>(type)) {
    // TileType -> !pto.tile<RxCxDTYPE>
    // Extract rows and cols from shape
    int rows = 32;   // Default
    int cols = 128;  // Default

    if (tile_type->shape_.size() >= 2) {
      if (auto rows_const = As<ConstInt>(tile_type->shape_[0])) {
        rows = rows_const->value_;
      }
      if (auto cols_const = As<ConstInt>(tile_type->shape_[1])) {
        cols = cols_const->value_;
      }
    }

    std::string dtype_str = DataTypeToPTOType(tile_type->dtype_);
    return "!pto.tile<" + std::to_string(rows) + "x" + std::to_string(cols) + "x" + dtype_str + ">";
  } else if (auto scalar_type = As<ScalarType>(type)) {
    // ScalarType -> i32, f32, u1, etc.
    return DataTypeToPTOType(scalar_type->dtype_);
  } else if (auto tensor_type = As<TensorType>(type)) {
    // TensorType with memref -> !pto.memref<gm,...,f32>
    // For now, assume global memory space
    std::string space = "gm";  // Default to global memory
    std::string dtype_str = DataTypeToPTOType(tensor_type->dtype_);
    return "!pto.memref<" + space + ",...," + dtype_str + ">";
  } else {
    // Fallback for unknown types
    return "!pto.unknown";
  }
}

ExprPtr PTOCodegen::VisitExpr_(const CallPtr& op) { return IRMutator::VisitExpr_(op); }

StmtPtr PTOCodegen::VisitStmt_(const AssignStmtPtr& op) {
  // Generate PTO assembly for this assignment
  if (auto call = As<Call>(op->value_)) {
    // Skip block.alloc operations - they are handled in the declaration section
    if (call->op_->name_ == "block.alloc") {
      return op;
    }

    // Resolve physical tile name for the result variable
    std::string physical_result = ResolvePhysicalTile(op->var_->name_);
    std::string pto_instr = OpToPTOInstruction(call, physical_result);
    EmitLine(pto_instr);
  } else if (auto const_int = As<ConstInt>(op->value_)) {
    // Generate LI instruction for integer constant: LI %var:type, value
    std::string var_type = "i32";  // Default
    if (op->var_->GetType()) {
      if (auto scalar_type = As<ScalarType>(op->var_->GetType())) {
        var_type = DataTypeToPTOType(scalar_type->dtype_);
      }
    }
    EmitLine("LI %" + op->var_->name_ + ":" + var_type + ", " + std::to_string(const_int->value_));
  } else if (auto const_float = As<ConstFloat>(op->value_)) {
    // Generate LI instruction for float constant: LI %var:type, value
    std::string var_type = "f32";  // Default
    if (op->var_->GetType()) {
      if (auto scalar_type = As<ScalarType>(op->var_->GetType())) {
        var_type = DataTypeToPTOType(scalar_type->dtype_);
      }
    }
    EmitLine("LI %" + op->var_->name_ + ":" + var_type + ", " + FormatFloat(const_float->value_));
  } else if (auto ge = As<Ge>(op->value_)) {
    EmitComparisonInstruction(op->var_, ge->left_, ge->right_, "GE");
  } else if (auto gt = As<Gt>(op->value_)) {
    EmitComparisonInstruction(op->var_, gt->left_, gt->right_, "GT");
  } else if (auto le = As<Le>(op->value_)) {
    EmitComparisonInstruction(op->var_, le->left_, le->right_, "LE");
  } else if (auto lt = As<Lt>(op->value_)) {
    EmitComparisonInstruction(op->var_, lt->left_, lt->right_, "LT");
  } else if (auto eq = As<Eq>(op->value_)) {
    EmitComparisonInstruction(op->var_, eq->left_, eq->right_, "EQ");
  } else if (auto ne = As<Ne>(op->value_)) {
    EmitComparisonInstruction(op->var_, ne->left_, ne->right_, "NE");
  }

  // Return original node (no children to visit in AssignStmt value expressions)
  return op;
}

StmtPtr PTOCodegen::VisitStmt_(const ForStmtPtr& op) {
  // Generate FOR instruction: FOR %var:type, start:type, stop:type, step:type
  std::string loop_var = op->loop_var_->name_;
  std::string start = ExtractVarName(op->start_);
  std::string stop = ExtractVarName(op->stop_);
  std::string step = ExtractVarName(op->step_);

  // Get type for loop variable - default to i32 for integers
  std::string var_type = "i32";  // Default to i32 type
  if (op->loop_var_->GetType()) {
    if (auto scalar_type = As<ScalarType>(op->loop_var_->GetType())) {
      var_type = DataTypeToPTOType(scalar_type->dtype_);
    }
  }

  // Build FOR line
  std::ostringstream oss;
  oss << "FOR %" << loop_var << ":" << var_type << ", ";

  // Add start (with type)
  if (auto const_int = As<ConstInt>(op->start_)) {
    oss << const_int->value_ << ":" << var_type;
  } else {
    oss << "%" << start << ":" << var_type;
  }
  oss << ", ";

  // Add stop (with type)
  if (auto const_int = As<ConstInt>(op->stop_)) {
    oss << const_int->value_ << ":" << var_type;
  } else {
    oss << "%" << stop << ":" << var_type;
  }
  oss << ", ";

  // Add step (with type)
  if (auto const_int = As<ConstInt>(op->step_)) {
    oss << const_int->value_ << ":" << var_type;
  } else {
    oss << "%" << step << ":" << var_type;
  }

  EmitLine(oss.str());

  // Increase indent and process loop body
  indent_level_++;
  VisitStmt(op->body_);
  indent_level_--;

  // Generate ENDFOR
  EmitLine("ENDFOR");

  // Return original node (we already visited children manually)
  return op;
}

StmtPtr PTOCodegen::VisitStmt_(const IfStmtPtr& op) {
  // Generate IF instruction: IF %cond:type
  std::string condition = ExtractVarName(op->condition_);

  // Get condition type
  std::string cond_type = "u1";  // Default to boolean
  if (op->condition_->GetType()) {
    if (auto scalar_type = As<ScalarType>(op->condition_->GetType())) {
      cond_type = DataTypeToPTOType(scalar_type->dtype_);
    }
  }

  EmitLine("IF %" + condition + ":" + cond_type);

  // Process then branch
  indent_level_++;
  VisitStmt(op->then_body_);
  indent_level_--;

  // Process else branch if present
  if (op->else_body_.has_value()) {
    EmitLine("ELSE");
    indent_level_++;
    VisitStmt(op->else_body_.value());
    indent_level_--;
  }

  // Generate ENDIF
  EmitLine("ENDIF");

  // Return original node (we already visited children manually)
  return op;
}

void PTOCodegen::EmitComparisonInstruction(const VarPtr& result_var, const ExprPtr& left,
                                           const ExprPtr& right, const std::string& cmp_op) {
  // Get operand names
  std::string left_name = ExtractVarName(left);
  std::string right_name = ExtractVarName(right);

  // Get operand types
  std::string result_type = "u1";  // Comparison results are always boolean (u1)
  std::string left_type = "i32";   // Default type
  std::string right_type = "i32";  // Default type

  if (left->GetType()) {
    if (auto scalar_type = As<ScalarType>(left->GetType())) {
      left_type = DataTypeToPTOType(scalar_type->dtype_);
    }
  }

  if (right->GetType()) {
    if (auto scalar_type = As<ScalarType>(right->GetType())) {
      right_type = DataTypeToPTOType(scalar_type->dtype_);
    }
  }

  // Build CMP instruction: CMP %result:u1, %left:type, %right:type, OP
  std::ostringstream oss;
  oss << "CMP %" << result_var->name_ << ":" << result_type << ", ";

  // Add left operand with type
  if (auto const_int = As<ConstInt>(left)) {
    oss << const_int->value_ << ":" << left_type;
  } else {
    oss << "%" << left_name << ":" << left_type;
  }
  oss << ", ";

  // Add right operand with type
  if (auto const_int = As<ConstInt>(right)) {
    oss << const_int->value_ << ":" << right_type;
  } else {
    oss << "%" << right_name << ":" << right_type;
  }

  oss << ", " << cmp_op;

  EmitLine(oss.str());
}

std::string PTOCodegen::FormatFloat(double value) {
  std::ostringstream oss;
  oss << std::defaultfloat << value;
  std::string result = oss.str();
  // Ensure decimal point exists (e.g., "2" -> "2.0")
  // Skip if scientific notation is used (contains 'e' or 'E')
  if (result.find('.') == std::string::npos && result.find('e') == std::string::npos &&
      result.find('E') == std::string::npos) {
    result += ".0";
  }
  return result;
}

StmtPtr PTOCodegen::VisitStmt_(const SeqStmtsPtr& op) {
  // Process each statement in sequence
  for (const auto& stmt : op->stmts_) {
    VisitStmt(stmt);
  }

  // Return original node (we already visited children manually)
  return op;
}

}  // namespace ir
}  // namespace pypto
