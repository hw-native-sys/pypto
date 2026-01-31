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

#include "pypto/codegen/pto/pto_codegen.h"

#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::AssignStmtPtr;
using ir::BinaryExprPtr;
using ir::CallPtr;
using ir::ConstFloatPtr;
using ir::ConstIntPtr;
using ir::EvalStmtPtr;
using ir::ExprPtr;
using ir::Function;
using ir::FunctionPtr;
using ir::FunctionType;
using ir::IRVisitor;
using ir::IterArgPtr;
using ir::MemRefPtr;
using ir::OpStmtsPtr;
using ir::ProgramPtr;
using ir::ScalarType;
using ir::SeqStmtsPtr;
using ir::StmtPtr;
using ir::TensorType;
using ir::TileType;
using ir::UnaryExprPtr;
using ir::VarPtr;

// Helper function to convert DataType to MLIR type string
static std::string DataTypeToMLIR(::pypto::DataType dtype) {
  if (dtype == ::pypto::DataType::FP32) {
    return "f32";
  } else if (dtype == ::pypto::DataType::FP16) {
    return "f16";
  } else if (dtype == ::pypto::DataType::BF16) {
    return "bf16";
  } else if (dtype == ::pypto::DataType::INT32) {
    return "i32";
  } else if (dtype == ::pypto::DataType::INT64) {
    return "i64";
  } else if (dtype == ::pypto::DataType::INT8) {
    return "i8";
  } else if (dtype == ::pypto::DataType::UINT8) {
    return "ui8";
  } else {
    return "f32";  // Default
  }
}

// Helper function to convert MemorySpace to MLIR string
static std::string MemorySpaceToMLIR(ir::MemorySpace space) {
  if (space == ir::MemorySpace::DDR) {
    return "ddr";
  } else if (space == ir::MemorySpace::UB) {
    return "ub";
  } else if (space == ir::MemorySpace::L1) {
    return "l1";
  } else if (space == ir::MemorySpace::L0A) {
    return "l0a";
  } else if (space == ir::MemorySpace::L0B) {
    return "l0b";
  } else if (space == ir::MemorySpace::L0C) {
    return "l0c";
  } else {
    return "ub";  // Default
  }
}

// Visitor to collect all MemRef objects from TileType variables
class MemRefCollectorVisitor : public IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  [[nodiscard]] const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }

  void VisitExpr_(const VarPtr& op) override {
    // Check if this variable has a TileType with MemRef
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value());
    }
  }

  void VisitExpr_(const IterArgPtr& op) override {
    // Check if this iteration argument has a TileType with MemRef
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value());
    }
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const ir::MemRef*> seen_ptrs_;  // Track raw MemRef pointers to avoid duplicates

  void AddMemRefIfUnique(const MemRefPtr& memref) {
    // Use raw pointer address to check uniqueness (same shared_ptr)
    const ir::MemRef* raw_ptr = memref.get();
    if (seen_ptrs_.find(raw_ptr) == seen_ptrs_.end()) {
      memrefs_.push_back(memref);
      seen_ptrs_.insert(raw_ptr);
    }
  }
};

/**
 * @brief PTO MLIR code generator implementation
 *
 * Generates MLIR code in PTO-ISA format from PyPTO IR.
 * Uses two-pass approach:
 * 1. First pass: collect all MemRefs and build variable mappings
 * 2. Second pass: generate MLIR code with proper variable references
 */
class PTOMLIRCodegen : public IRVisitor {
 public:
  PTOMLIRCodegen() = default;

  std::string Generate(const ProgramPtr& program) {
    stream_.str("");
    stream_.clear();

    // Generate module wrapper
    stream_ << "module {\n";

    // Generate each function
    for (const auto& [gvar, func] : program->functions_) {
      GenerateFunction(func);
    }

    stream_ << "}\n";
    return stream_.str();
  }

 private:
  std::ostringstream stream_;
  std::ostringstream constants_section_;
  std::ostringstream body_section_;
  int indent_level_ = 0;

  // Variable mappings
  std::map<std::string, std::string> var_to_mlir_;           // IR variable name -> MLIR name
  std::map<std::string, std::string> tensor_to_view_;        // tensor parameter -> tensor_view name
  std::map<const ir::MemRef*, std::string> memref_to_mlir_;  // MemRef pointer -> tile_buf MLIR name
  std::map<std::string, const ir::MemRef*> var_to_memref_;   // Variable name -> MemRef (for reverse lookup)
  std::set<int64_t> emitted_constants_;                      // Already emitted constant values
  std::set<double> emitted_float_constants_;                 // Float constants
  std::map<double, std::string> float_const_names_;          // Float value -> constant name

  int temp_counter_ = 0;

  // Current function context
  FunctionPtr current_function_;
  std::string current_tile_view_;   // Track last generated tile_view for tload
  std::string current_result_buf_;  // Track result tile_buf for compute operations

  // Helper: Get indent string
  std::string GetIndent() const { return std::string(indent_level_ * 2, ' '); }

  // Helper: Generate new temporary variable name
  std::string NewTemp() { return "%" + std::to_string(temp_counter_++); }

  // Helper: Get or emit index constant
  std::string GetOrEmitIndexConstant(int64_t value) {
    std::string name = "%c" + std::to_string(value);
    if (emitted_constants_.find(value) == emitted_constants_.end()) {
      constants_section_ << GetIndent() << name << " = arith.constant " << value << " : index\n";
      emitted_constants_.insert(value);
    }
    return name;
  }

  // Helper: Get or emit float constant
  std::string GetOrEmitFloatConstant(double value, const std::string& dtype = "f32") {
    if (emitted_float_constants_.find(value) == emitted_float_constants_.end()) {
      std::string name = "%cst";
      if (!emitted_float_constants_.empty()) {
        name += "_" + std::to_string(emitted_float_constants_.size());
      }

      std::ostringstream val_str;
      val_str << std::scientific << std::setprecision(6) << value;

      constants_section_ << GetIndent() << name << " = arith.constant " << val_str.str() << " : " << dtype
                         << "\n";
      emitted_float_constants_.insert(value);
      float_const_names_[value] = name;
      return name;
    }
    return float_const_names_[value];
  }

  // Helper: Extract constant int value from expression
  int64_t GetConstIntValue(const ExprPtr& expr) {
    if (auto const_int = As<ir::ConstInt>(expr)) {
      return const_int->value_;
    }
    LOG_ERROR << "Expected ConstInt expression";
    return 0;
  }

  // Helper: Get tile_buf name for a MemRef
  std::string GetTileBufForMemRef(const MemRefPtr& memref) {
    auto it = memref_to_mlir_.find(memref.get());
    INTERNAL_CHECK(it != memref_to_mlir_.end()) << "MemRef not found in mapping";
    return it->second;
  }

  // Helper: Get or create tensor_view for a tensor parameter
  std::string GetOrCreateTensorView(const VarPtr& tensor_param) {
    auto it = tensor_to_view_.find(tensor_param->name_);
    INTERNAL_CHECK(it != tensor_to_view_.end())
        << "Tensor view not found for parameter: " << tensor_param->name_;
    return it->second;
  }

  // Build variable to MemRef mapping
  void BuildVarToMemRefMapping(const FunctionPtr& func) {
    // Visitor to map variables to their MemRefs
    class VarMemRefMapper : public IRVisitor {
     public:
      std::map<std::string, const ir::MemRef*>& var_to_memref;

      explicit VarMemRefMapper(std::map<std::string, const ir::MemRef*>& mapping) : var_to_memref(mapping) {}

      void VisitStmt_(const AssignStmtPtr& op) override {
        if (auto tile_type = As<TileType>(op->var_->GetType())) {
          if (tile_type->memref_.has_value()) {
            var_to_memref[op->var_->name_] = tile_type->memref_.value().get();
          }
        }
        IRVisitor::VisitStmt_(op);
      }
    };

    VarMemRefMapper mapper(var_to_memref_);
    if (func->body_) {
      mapper.VisitStmt(func->body_);
    }
  }

  // Generate function and MLIR code
  void GenerateFunction(const FunctionPtr& func) {
    current_function_ = func;
    temp_counter_ = 0;
    var_to_mlir_.clear();
    tensor_to_view_.clear();
    memref_to_mlir_.clear();
    var_to_memref_.clear();
    emitted_constants_.clear();
    emitted_float_constants_.clear();
    float_const_names_.clear();
    constants_section_.str("");
    constants_section_.clear();
    body_section_.str("");
    body_section_.clear();

    // Build variable to MemRef mapping
    BuildVarToMemRefMapping(func);

    // Collect all MemRefs for alloc_tile generation
    MemRefCollectorVisitor collector;
    if (func->body_) {
      collector.VisitStmt(func->body_);
    }

    // Pre-assign tile_buf names to MemRefs
    for (const auto& memref : collector.GetMemRefs()) {
      std::string tile_buf = NewTemp();
      memref_to_mlir_[memref.get()] = tile_buf;
    }

    // Emit function signature
    stream_ << "  func.func @" << func->name_ << "(";

    // First, map all parameters to their %argN names
    std::set<std::string> param_names;
    for (size_t i = 0; i < func->params_.size(); i++) {
      if (i > 0) stream_ << ", ";
      auto& param = func->params_[i];
      std::string arg_name = "%arg" + std::to_string(i);
      stream_ << arg_name << ": ";

      // Map parameter name to MLIR name
      var_to_mlir_[param->name_] = arg_name;
      param_names.insert(param->name_);

      // Generate type based on parameter type
      if (auto tensor_type = As<TensorType>(param->GetType())) {
        stream_ << "!pto.ptr<" << DataTypeToMLIR(tensor_type->dtype_) << ">";
      } else if (auto scalar_type = As<ScalarType>(param->GetType())) {
        stream_ << DataTypeToMLIR(scalar_type->dtype_);
      } else {
        stream_ << "!pto.ptr<f32>";  // Default
      }
    }

    stream_ << ") {\n";
    indent_level_++;

    // Now map internal variables (non-parameters) to their tile_bufs
    for (const auto& [var_name, memref_ptr] : var_to_memref_) {
      // Skip parameters - they should keep their %argN mapping
      if (param_names.find(var_name) == param_names.end()) {
        var_to_mlir_[var_name] = memref_to_mlir_[memref_ptr];
      }
    }

    // Pre-assign tensor_view names for tensor parameters (needed before body traversal)
    // Also pre-collect constants needed for tensor_views
    for (size_t i = 0; i < func->params_.size(); i++) {
      auto& param = func->params_[i];
      if (auto tensor_type = As<TensorType>(param->GetType())) {
        std::string tensor_view = NewTemp();
        tensor_to_view_[param->name_] = tensor_view;

        // Pre-collect shape and stride constants
        for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
          int64_t dim = GetConstIntValue(tensor_type->shape_[j]);
          GetOrEmitIndexConstant(dim);
        }
        // Stride constants
        if (tensor_type->shape_.size() == 2) {
          int64_t dim1 = GetConstIntValue(tensor_type->shape_[1]);
          GetOrEmitIndexConstant(dim1);
          GetOrEmitIndexConstant(1);
        } else if (tensor_type->shape_.size() == 1) {
          GetOrEmitIndexConstant(1);
        }
      }
    }

    // Generate body to collect constants and operations
    auto saved_stream = std::move(stream_);
    stream_ = std::move(body_section_);

    if (func->body_) {
      VisitStmt(func->body_);
    }

    std::string body_content = stream_.str();
    stream_ = std::move(saved_stream);

    // Now emit in correct order:
    // 1. Constants
    stream_ << constants_section_.str();

    // 2. make_tensor_view for tensor parameters
    EmitMakeTensorViews(func);

    // 3. alloc_tile for all MemRefs
    EmitAllocTiles(func, collector.GetMemRefs());

    // 4. Function body
    stream_ << body_content;

    // 5. return statement
    stream_ << GetIndent() << "return\n";

    indent_level_--;
    stream_ << "  }\n";
  }

  // Emit make_tensor_view for all tensor parameters
  void EmitMakeTensorViews(const FunctionPtr& func) {
    for (size_t i = 0; i < func->params_.size(); i++) {
      auto& param = func->params_[i];
      if (auto tensor_type = As<TensorType>(param->GetType())) {
        // tensor_view name was already assigned in EmitFunction
        std::string tensor_view = tensor_to_view_[param->name_];

        // Generate make_tensor_view
        stream_ << GetIndent() << tensor_view << " = pto.make_tensor_view ";
        stream_ << "%arg" << i;

        // shape
        stream_ << ", shape = [";
        for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
          if (j > 0) stream_ << ", ";
          int64_t dim = GetConstIntValue(tensor_type->shape_[j]);
          stream_ << GetOrEmitIndexConstant(dim);
        }
        stream_ << "]";

        // strides (row-major: [N, 1] for 2D)
        stream_ << " strides = [";
        if (tensor_type->shape_.size() == 2) {
          int64_t dim1 = GetConstIntValue(tensor_type->shape_[1]);
          stream_ << GetOrEmitIndexConstant(dim1) << ", " << GetOrEmitIndexConstant(1);
        } else if (tensor_type->shape_.size() == 1) {
          stream_ << GetOrEmitIndexConstant(1);
        }
        stream_ << "]";

        // Result type
        stream_ << " : !pto.tensor_view<" << tensor_type->shape_.size() << "x";
        stream_ << DataTypeToMLIR(tensor_type->dtype_) << ">\n";
      }
    }
  }

  // Emit alloc_tile for all MemRefs
  void EmitAllocTiles(const FunctionPtr& func, const std::vector<MemRefPtr>& memrefs) {
    for (const auto& memref : memrefs) {
      std::string tile_buf = memref_to_mlir_[memref.get()];

      // Get memory space
      std::string loc = MemorySpaceToMLIR(memref->memory_space_);

      // TODO(yifanlin): Get actual dimensions from TileType - for now use 32x32
      stream_ << GetIndent() << tile_buf << " = pto.alloc_tile : <loc=" << loc;
      stream_ << ", dtype=f32, rows=32, cols=32, v_row=32, v_col=32";
      stream_ << ", blayout=row_major, slayout=none_box, fractal=512, pad=0>\n";
    }
  }

  // Statement visitors
  void VisitStmt_(const AssignStmtPtr& op) override {
    // Check if this is an assignment from block.load
    if (auto call = As<ir::Call>(op->value_)) {
      if (call->op_->name_ == "block.load") {
        // Get the tile_buf for the result variable
        std::string tile_buf;
        if (auto tile_type = As<TileType>(op->var_->GetType())) {
          if (tile_type->memref_.has_value()) {
            tile_buf = GetTileBufForMemRef(tile_type->memref_.value());
          }
        }

        // Generate subview
        EmitBlockLoadSubview(call);

        // Generate tload with the result tile_buf
        if (!tile_buf.empty() && !current_tile_view_.empty()) {
          auto tensor = As<ir::Var>(call->args_[0]);
          auto tensor_type = As<TensorType>(tensor->GetType());
          int64_t height = GetConstIntValue(call->args_[3]);
          int64_t width = GetConstIntValue(call->args_[4]);
          std::string dtype_str = DataTypeToMLIR(tensor_type->dtype_);

          stream_ << GetIndent() << "pto.tload ins(" << current_tile_view_;
          stream_ << " : !pto.tile_view<" << height << "x" << width << "x" << dtype_str << ">) outs(";
          stream_ << tile_buf << " : !pto.tile_buf<loc=ub, dtype=" << dtype_str;
          stream_ << ", rows=" << height << ", cols=" << width;
          stream_ << ", v_row=" << height << ", v_col=" << width;
          stream_ << ", blayout=row_major, slayout=none_box, fractal=512, pad=0>)\n";

          current_tile_view_.clear();
        }
        return;
      } else if (call->op_->name_ == "block.mul" || call->op_->name_ == "block.add" ||
                 call->op_->name_ == "block.adds") {
        // Get result tile_buf for compute operations
        std::string result_buf;
        if (auto tile_type = As<TileType>(op->var_->GetType())) {
          if (tile_type->memref_.has_value()) {
            result_buf = GetTileBufForMemRef(tile_type->memref_.value());
          }
        }

        // Store result_buf in context for operation emission
        current_result_buf_ = result_buf;

        // Generate the compute operation
        VisitExpr(op->value_);

        current_result_buf_.clear();
        return;
      }
    }

    // For other expressions, just visit them
    VisitExpr(op->value_);
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const OpStmtsPtr& op) override {
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const EvalStmtPtr& op) override { VisitExpr(op->expr_); }

  // Expression visitors
  void VisitExpr_(const CallPtr& op) override {
    const std::string& op_name = op->op_->name_;

    if (op_name == "block.store") {
      EmitBlockStore(op);
    } else if (op_name == "block.mul") {
      EmitBlockMul(op);
    } else if (op_name == "block.add") {
      EmitBlockAdd(op);
    } else if (op_name == "block.adds") {
      EmitBlockAdds(op);
    }
    // block.load is handled in AssignStmt visitor
  }

  // Emit subview part of block.load (tload is emitted in AssignStmt)
  void EmitBlockLoadSubview(const CallPtr& op) {
    // block.load(tensor, row_offset, col_offset, height, width, target_memory=1)
    auto tensor = As<ir::Var>(op->args_[0]);
    INTERNAL_CHECK(tensor) << "block.load first argument must be a Var";

    int64_t row_off = GetConstIntValue(op->args_[1]);
    int64_t col_off = GetConstIntValue(op->args_[2]);
    int64_t height = GetConstIntValue(op->args_[3]);
    int64_t width = GetConstIntValue(op->args_[4]);

    auto tensor_type = As<TensorType>(tensor->GetType());
    INTERNAL_CHECK(tensor_type) << "block.load tensor argument must have TensorType";

    // Get tensor_view for the tensor parameter
    std::string tensor_view = GetOrCreateTensorView(tensor);
    std::string dtype_str = DataTypeToMLIR(tensor_type->dtype_);

    // Generate subview
    std::string tile_view = NewTemp();
    stream_ << GetIndent() << tile_view << " = pto.subview " << tensor_view;
    stream_ << ", offsets = [" << GetOrEmitIndexConstant(row_off) << ", ";
    stream_ << GetOrEmitIndexConstant(col_off) << "]";
    stream_ << ", sizes = [" << GetOrEmitIndexConstant(height) << ", ";
    stream_ << GetOrEmitIndexConstant(width) << "]";
    stream_ << " : !pto.tensor_view<2x" << dtype_str << "> -> !pto.tile_view<";
    stream_ << height << "x" << width << "x" << dtype_str << ">\n";

    // Store tile_view name for later tload generation
    current_tile_view_ = tile_view;
  }

  // Emit block.store -> subview + tstore
  void EmitBlockStore(const CallPtr& op) {
    // block.store(tile, row_offset, col_offset, height, width, output_tensor)
    auto tile = As<ir::Var>(op->args_[0]);
    int64_t row_off = GetConstIntValue(op->args_[1]);
    int64_t col_off = GetConstIntValue(op->args_[2]);
    int64_t height = GetConstIntValue(op->args_[3]);
    int64_t width = GetConstIntValue(op->args_[4]);
    auto output_tensor = As<ir::Var>(op->args_[5]);

    auto tensor_type = As<TensorType>(output_tensor->GetType());
    std::string dtype_str = DataTypeToMLIR(tensor_type->dtype_);

    // 1. Get tensor_view for the output tensor
    std::string tensor_view = GetOrCreateTensorView(output_tensor);

    // 2. Generate subview
    std::string tile_view = NewTemp();
    stream_ << GetIndent() << tile_view << " = pto.subview " << tensor_view;
    stream_ << ", offsets = [" << GetOrEmitIndexConstant(row_off) << ", ";
    stream_ << GetOrEmitIndexConstant(col_off) << "]";
    stream_ << ", sizes = [" << GetOrEmitIndexConstant(height) << ", ";
    stream_ << GetOrEmitIndexConstant(width) << "]";
    stream_ << " : !pto.tensor_view<2x" << dtype_str << "> -> !pto.tile_view<";
    stream_ << height << "x" << width << "x" << dtype_str << ">\n";

    // 3. Get tile_buf from tile variable
    std::string tile_buf = var_to_mlir_[tile->name_];

    // 4. Generate tstore
    stream_ << GetIndent() << "pto.tstore ins(" << tile_buf;
    stream_ << " : !pto.tile_buf<loc=ub, dtype=" << dtype_str << ", rows=" << height;
    stream_ << ", cols=" << width << ", v_row=" << height << ", v_col=" << width;
    stream_ << ", blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(";
    stream_ << tile_view << " : !pto.tile_view<" << height << "x" << width << "x" << dtype_str << ">)\n";
  }

  // Emit block.mul -> tmul
  void EmitBlockMul(const CallPtr& op) {
    // block.mul(lhs, rhs)
    auto lhs = As<ir::Var>(op->args_[0]);
    auto rhs = As<ir::Var>(op->args_[1]);

    std::string lhs_buf = var_to_mlir_[lhs->name_];
    std::string rhs_buf = var_to_mlir_[rhs->name_];

    // Use current_result_buf_ if available, otherwise use placeholder
    std::string result_buf = current_result_buf_.empty() ? "RESULT_BUF" : current_result_buf_;

    stream_ << GetIndent() << "pto.tmul ins(" << lhs_buf;
    stream_ << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>, " << rhs_buf;
    stream_ << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(";
    stream_ << result_buf;
    stream_ << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>)\n";
  }

  // Emit block.add -> taddc (3-input add)
  void EmitBlockAdd(const CallPtr& op) {
    // block.add(a, b, c) -> a + b + c
    auto a = As<ir::Var>(op->args_[0]);
    auto b = As<ir::Var>(op->args_[1]);
    auto c = As<ir::Var>(op->args_[2]);

    std::string a_buf = var_to_mlir_[a->name_];
    std::string b_buf = var_to_mlir_[b->name_];
    std::string c_buf = var_to_mlir_[c->name_];

    // Use current_result_buf_ if available
    std::string result_buf = current_result_buf_.empty() ? "RESULT_BUF" : current_result_buf_;

    stream_ << GetIndent() << "pto.taddc ins(" << a_buf << ", " << b_buf << ", " << c_buf;
    stream_ << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>, ";
    stream_ << "!pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>, ";
    stream_ << "!pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(";
    stream_ << result_buf << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>)\n";
  }

  // Emit block.adds -> tadds (tile + scalar)
  void EmitBlockAdds(const CallPtr& op) {
    // block.adds(tile, scalar)
    auto tile = As<ir::Var>(op->args_[0]);

    std::string tile_buf = var_to_mlir_[tile->name_];

    // Get scalar value and emit as constant
    double scalar_val = 3.14;
    if (auto const_float = As<ir::ConstFloat>(op->args_[1])) {
      scalar_val = const_float->value_;
    } else if (auto const_int = As<ir::ConstInt>(op->args_[1])) {
      scalar_val = static_cast<double>(const_int->value_);
    }

    // Get dtype from scalar argument
    std::string scalar_type = "f32";
    if (auto scalar_t = As<ScalarType>(op->args_[1]->GetType())) {
      scalar_type = DataTypeToMLIR(scalar_t->dtype_);
    }

    std::string scalar_const = GetOrEmitFloatConstant(scalar_val, scalar_type);

    // Use current_result_buf_ if available
    std::string result_buf = current_result_buf_.empty() ? "RESULT_BUF" : current_result_buf_;

    stream_ << GetIndent() << "pto.tadds ins(" << tile_buf << ", " << scalar_const;
    stream_ << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>, " << scalar_type << ") outs(";
    stream_ << result_buf;
    stream_ << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    stream_ << "blayout=row_major, slayout=none_box, fractal=512, pad=0>)\n";
  }
};

// Public API implementation
std::string PTOCodegen::Generate(const ir::ProgramPtr& program) {
  PTOMLIRCodegen codegen;
  return codegen.Generate(program);
}

}  // namespace codegen
}  // namespace pypto
