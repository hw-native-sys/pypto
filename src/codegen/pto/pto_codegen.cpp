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

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::AssignStmtPtr;
using ir::BinaryExprPtr;
using ir::CallPtr;
using ir::EvalStmtPtr;
using ir::ExprPtr;
using ir::ForStmtPtr;
using ir::FunctionPtr;
using ir::IfStmtPtr;
using ir::MemRefPtr;
using ir::ProgramPtr;
using ir::ScalarType;
using ir::StmtPtr;
using ir::TensorType;
using ir::TileType;
using ir::VarPtr;
using ir::YieldStmtPtr;

// Helper function to convert DataType to MLIR type string
static std::string DataTypeToMLIRImpl(::pypto::DataType dtype) {
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
    throw pypto::ValueError("Invalid DataType value");
  }
}

// Helper function to convert MemorySpace to PTO address space string
static std::string MemorySpaceToMLIR(ir::MemorySpace space) {
  if (space == ir::MemorySpace::DDR) {
    return "gm";
  } else if (space == ir::MemorySpace::Vec) {
    return "vec";
  } else if (space == ir::MemorySpace::Mat) {
    return "mat";
  } else if (space == ir::MemorySpace::Left) {
    return "left";
  } else if (space == ir::MemorySpace::Right) {
    return "right";
  } else if (space == ir::MemorySpace::Acc) {
    return "acc";
  } else {
    throw pypto::ValueError("Invalid MemorySpace value");
  }
}

// Visitor to collect all MemRef objects from TileType variables
class MemRefCollectorVisitor : public ir::IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  [[nodiscard]] const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }
  [[nodiscard]] const std::map<const ir::MemRef*, std::shared_ptr<const TileType>>& GetMemRefTileTypes()
      const {
    return memref_tile_types_;
  }

  void VisitExpr_(const VarPtr& op) override {
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value(), tile_type);
    }
  }

  void VisitExpr_(const ir::IterArgPtr& op) override {
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value(), tile_type);
    }
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const ir::MemRef*> seen_ptrs_;
  std::map<const ir::MemRef*, std::shared_ptr<const TileType>> memref_tile_types_;

  void AddMemRefIfUnique(const MemRefPtr& memref, const std::shared_ptr<const TileType>& tile_type) {
    const ir::MemRef* raw_ptr = memref.get();
    if (seen_ptrs_.find(raw_ptr) == seen_ptrs_.end()) {
      memrefs_.push_back(memref);
      seen_ptrs_.insert(raw_ptr);
      memref_tile_types_[raw_ptr] = tile_type;
    }
  }
};

// ========================================================================
// Constructors
// ========================================================================

PTOCodegen::PTOCodegen() : backend_(backend::GetBackend()) {
  auto type = backend::GetBackendType();
  CHECK(type == backend::BackendType::PTO)
      << "PTOCodegen requires PTO backend, but " << (type == backend::BackendType::CCE ? "CCE" : "unknown")
      << " is configured";
}

PTOCodegen::PTOCodegen(const backend::Backend* backend) : backend_(backend) {
  CHECK(backend != nullptr) << "Backend cannot be null";
}

// ========================================================================
// Generate entry and GenerateFunction
// ========================================================================

std::string PTOCodegen::Generate(const ProgramPtr& program) {
  stream_.str("");
  stream_.clear();
  constants_section_.str("");
  constants_section_.clear();
  body_section_.str("");
  body_section_.clear();

  stream_ << "module {\n";

  for (const auto& [gvar, func] : program->functions_) {
    if (func->func_type_ == ir::FunctionType::Orchestration) {
      throw pypto::ValueError(
          "PTO backend does not support Orchestration functions. "
          "Function '" +
          func->name_ + "' is marked as Orchestration. ");
    }
    GenerateFunction(func);
  }

  stream_ << "}\n";
  return stream_.str();
}

void PTOCodegen::GenerateFunction(const FunctionPtr& func) {
  current_function_ = func;
  temp_counter_ = 0;
  var_to_mlir_.clear();
  tensor_to_view_.clear();
  memref_to_mlir_.clear();
  var_to_memref_.clear();
  memref_to_tile_type_.clear();
  emitted_constants_.clear();
  emitted_float_constants_.clear();
  float_const_names_.clear();
  constants_section_.str("");
  constants_section_.clear();
  body_section_.str("");
  body_section_.clear();

  BuildVarToMemRefMapping(func);

  MemRefCollectorVisitor collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }

  for (const auto& memref : collector.GetMemRefs()) {
    std::string tile_buf = NewTemp();
    memref_to_mlir_[memref.get()] = tile_buf;
  }
  memref_to_tile_type_ = collector.GetMemRefTileTypes();

  stream_ << "  func.func @" << func->name_ << "(";

  std::set<std::string> param_names;
  for (size_t i = 0; i < func->params_.size(); i++) {
    if (i > 0) stream_ << ", ";
    const auto& param = func->params_[i];
    std::string arg_name = "%arg" + std::to_string(i);
    stream_ << arg_name << ": ";

    var_to_mlir_[param->name_] = arg_name;
    param_names.insert(param->name_);

    if (auto tensor_type = As<TensorType>(param->GetType())) {
      stream_ << "!pto.ptr<" << GetTypeString(tensor_type->dtype_) << ">";
    } else if (auto scalar_type = As<ScalarType>(param->GetType())) {
      stream_ << GetTypeString(scalar_type->dtype_);
    } else {
      stream_ << "!pto.ptr<f32>";
    }
  }

  stream_ << ") {\n";
  indent_level_++;

  for (const auto& [var_name, memref_ptr] : var_to_memref_) {
    if (param_names.find(var_name) == param_names.end()) {
      var_to_mlir_[var_name] = memref_to_mlir_[memref_ptr];
    }
  }

  for (const auto& var : func->params_) {
    if (auto tensor_type = As<TensorType>(var->GetType())) {
      std::string tensor_view = NewTemp();
      tensor_to_view_[var->name_] = tensor_view;

      for (const auto& j : tensor_type->shape_) {
        int64_t dim = GetConstIntValue(j);
        GetOrEmitIndexConstant(dim);
      }
      if (tensor_type->shape_.size() == 2) {
        int64_t dim1 = GetConstIntValue(tensor_type->shape_[1]);
        GetOrEmitIndexConstant(dim1);
        GetOrEmitIndexConstant(1);
      } else if (tensor_type->shape_.size() == 1) {
        GetOrEmitIndexConstant(1);
      }
    }
  }

  auto saved_stream = std::move(stream_);
  stream_ = std::move(body_section_);

  if (func->body_) {
    VisitStmt(func->body_);
  }

  std::string body_content = stream_.str();
  stream_ = std::move(saved_stream);

  stream_ << constants_section_.str();
  EmitMakeTensorViews(func);
  EmitAllocTiles(func, collector.GetMemRefs());
  stream_ << body_content;
  stream_ << GetIndent() << "return\n";

  indent_level_--;
  stream_ << "  }\n";
}

void PTOCodegen::BuildVarToMemRefMapping(const FunctionPtr& func) {
  class VarMemRefMapper : public ir::IRVisitor {
   public:
    std::map<std::string, const ir::MemRef*>& var_to_memref;

    explicit VarMemRefMapper(std::map<std::string, const ir::MemRef*>& mapping) : var_to_memref(mapping) {}

    void VisitStmt_(const AssignStmtPtr& op) override {
      if (auto tile_type = As<TileType>(op->var_->GetType())) {
        if (tile_type->memref_.has_value()) {
          var_to_memref[op->var_->name_] = tile_type->memref_.value().get();
        }
      }
      ir::IRVisitor::VisitStmt_(op);
    }
  };

  VarMemRefMapper mapper(var_to_memref_);
  if (func->body_) {
    mapper.VisitStmt(func->body_);
  }
}

void PTOCodegen::EmitMakeTensorViews(const FunctionPtr& func) {
  for (size_t i = 0; i < func->params_.size(); i++) {
    const auto& param = func->params_[i];
    if (auto tensor_type = As<TensorType>(param->GetType())) {
      std::string tensor_view = tensor_to_view_[param->name_];

      stream_ << GetIndent() << tensor_view << " = pto.make_tensor_view ";
      stream_ << "%arg" << i;

      stream_ << ", shape = [";
      for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
        if (j > 0) stream_ << ", ";
        int64_t dim = GetConstIntValue(tensor_type->shape_[j]);
        stream_ << GetOrEmitIndexConstant(dim);
      }
      stream_ << "]";

      stream_ << " strides = [";
      if (tensor_type->shape_.size() == 2) {
        int64_t dim1 = GetConstIntValue(tensor_type->shape_[1]);
        stream_ << GetOrEmitIndexConstant(dim1) << ", " << GetOrEmitIndexConstant(1);
      } else if (tensor_type->shape_.size() == 1) {
        stream_ << GetOrEmitIndexConstant(1);
      }
      stream_ << "]";

      stream_ << " : !pto.tensor_view<";
      for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
        if (j > 0) stream_ << "x";
        stream_ << "?";
      }
      stream_ << "x" << GetTypeString(tensor_type->dtype_) << ">\n";
    }
  }
}

void PTOCodegen::EmitAllocTiles(const ir::FunctionPtr& func, const std::vector<ir::MemRefPtr>& memrefs) {
  (void)func;
  for (const auto& memref : memrefs) {
    std::string tile_buf = memref_to_mlir_[memref.get()];
    stream_ << GetIndent() << tile_buf << " = pto.alloc_tile : " << GetTileBufTypeString(memref.get())
            << "\n";
  }
}

// ========================================================================
// Private helpers
// ========================================================================

std::string PTOCodegen::GetIndent() const { return std::string(static_cast<size_t>(indent_level_) * 2, ' '); }

std::string PTOCodegen::GetOrEmitIndexConstant(int64_t value) {
  std::string name = "%c" + std::to_string(value);
  if (emitted_constants_.find(value) == emitted_constants_.end()) {
    constants_section_ << GetIndent() << name << " = arith.constant " << value << " : index\n";
    emitted_constants_.insert(value);
  }
  return name;
}

std::string PTOCodegen::GetTileBufForMemRef(const MemRefPtr& memref) {
  auto it = memref_to_mlir_.find(memref.get());
  INTERNAL_CHECK(it != memref_to_mlir_.end()) << "MemRef not found in mapping";
  return it->second;
}

// ========================================================================
// Statement visitors
// ========================================================================

void PTOCodegen::VisitStmt_(const AssignStmtPtr& op) {
  if (auto call = As<ir::Call>(op->value_)) {
    if (backend_ != nullptr && backend_->GetOpInfo(call->op_->name_) != nullptr) {
      std::string result_buf;
      std::shared_ptr<const TileType> result_tile_type;
      if (auto tile_type = As<TileType>(op->var_->GetType())) {
        if (tile_type->memref_.has_value()) {
          result_buf = GetTileBufForMemRef(tile_type->memref_.value());
        }
        result_tile_type = tile_type;
      }
      current_result_buf_ = result_buf;
      current_result_tile_type_ = result_tile_type;
      VisitExpr(op->value_);
      current_result_buf_.clear();
      current_result_tile_type_ = nullptr;
      return;
    }
  }

  VisitExpr(op->value_);
}

// ========================================================================
// Expression visitors
// ========================================================================

void PTOCodegen::VisitExpr_(const CallPtr& op) {
  const std::string& op_name = op->op_->name_;

  CHECK(backend_ != nullptr) << "Backend must not be null; use PTOCodegen(backend) or default backend";
  const auto* op_info = backend_->GetOpInfo(op_name);
  if (op_info == nullptr) {
    ThrowNoCodegenForCall(op_name);
  }
  std::string mlir_line = op_info->codegen_func(op, *this);
  if (!mlir_line.empty()) {
    Emit(mlir_line);
  }
}

// ========================================================================
// CodegenBase interface and PTO-specific helper methods
// ========================================================================

std::string PTOCodegen::GetCurrentResultTarget() const { return current_result_buf_; }

void PTOCodegen::Emit(const std::string& line) { stream_ << GetIndent() << line << "\n"; }

std::string PTOCodegen::GetExprAsCode(const ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    return GetVarName(var);
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return GetIndexConstant(const_int->value_);
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return GetOrEmitFloatConstant(const_float->value_, "f32");
  }

  // Fall back to visitor pattern for complex expressions (arithmetic, comparisons)
  current_expr_value_ = "";
  VisitExpr(expr);
  std::string result = current_expr_value_;
  current_expr_value_ = "";
  if (!result.empty()) {
    return result;
  }

  LOG_ERROR << "GetExprAsCode for unsupported expression type";
  return "";
}

std::string PTOCodegen::GetTypeString(const DataType& dtype) const { return DataTypeToMLIRImpl(dtype); }

std::string PTOCodegen::GetVarName(const VarPtr& var) {
  auto it = var_to_mlir_.find(var->name_);
  if (it != var_to_mlir_.end()) {
    return it->second;
  }
  auto memref_it = var_to_memref_.find(var->name_);
  if (memref_it != var_to_memref_.end()) {
    auto mlir_it = memref_to_mlir_.find(memref_it->second);
    if (mlir_it != memref_to_mlir_.end()) {
      return mlir_it->second;
    }
  }
  LOG_ERROR << "Variable " << var->name_ << " not found in MLIR mapping";
  return "";
}

std::string PTOCodegen::NewTemp() { return "%" + std::to_string(temp_counter_++); }

int64_t PTOCodegen::GetConstIntValue(const ExprPtr& expr) {
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return const_int->value_;
  }
  LOG_ERROR << "Expected ConstInt expression";
  return 0;
}

std::string PTOCodegen::GetOrCreateTensorView(const VarPtr& tensor_param) {
  auto it = tensor_to_view_.find(tensor_param->name_);
  INTERNAL_CHECK(it != tensor_to_view_.end())
      << "Tensor view not found for parameter: " << tensor_param->name_;
  return it->second;
}

std::string PTOCodegen::GetIndexConstant(int64_t val) { return GetOrEmitIndexConstant(val); }

std::string PTOCodegen::GetOrEmitFloatConstant(double value, const std::string& mlir_type) {
  if (emitted_float_constants_.find(value) == emitted_float_constants_.end()) {
    std::string name = "%cst";
    if (!emitted_float_constants_.empty()) {
      name += "_" + std::to_string(emitted_float_constants_.size());
    }

    std::ostringstream val_str;
    val_str << std::scientific << std::setprecision(6) << value;

    constants_section_ << GetIndent() << name << " = arith.constant " << val_str.str() << " : " << mlir_type
                       << "\n";
    emitted_float_constants_.insert(value);
    float_const_names_[value] = name;
    return name;
  }
  return float_const_names_[value];
}

std::string PTOCodegen::GetTensorViewTypeString(const ir::TensorType* tensor_type) const {
  std::ostringstream oss;
  oss << "!pto.tensor_view<";
  for (size_t i = 0; i < tensor_type->shape_.size(); i++) {
    if (i > 0) oss << "x";
    oss << "?";
  }
  oss << "x" << GetTypeString(tensor_type->dtype_) << ">";
  return oss.str();
}

std::string PTOCodegen::GetTileBufTypeString(const ir::MemRef* memref) const {
  std::string loc = MemorySpaceToMLIR(memref->memory_space_);

  // Get dtype and dimensions from the associated TileType
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;

  // Extract blayout, slayout, fractal, pad from TileView if available, otherwise use defaults
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::TilePad pad = ir::TilePad::null;

  auto tile_it = memref_to_tile_type_.find(memref);
  if (tile_it != memref_to_tile_type_.end()) {
    const auto& tile_type = tile_it->second;
    dtype_str = GetTypeString(tile_type->dtype_);
    if (tile_type->shape_.size() >= 2) {
      if (auto c0 = As<ir::ConstInt>(tile_type->shape_[0])) rows = c0->value_;
      if (auto c1 = As<ir::ConstInt>(tile_type->shape_[1])) cols = c1->value_;
    } else if (tile_type->shape_.size() == 1) {
      if (auto c0 = As<ir::ConstInt>(tile_type->shape_[0])) {
        rows = 1;
        cols = c0->value_;
      }
    }
    if (tile_type->tile_view_.has_value()) {
      const auto& tv = *tile_type->tile_view_;
      blayout = tv.blayout;
      slayout = tv.slayout;
      fractal = tv.fractal;
      pad = tv.pad;
    }
  }

  auto layout_to_str = [](ir::TileLayout layout) -> const char* {
    switch (layout) {
      case ir::TileLayout::none_box:
        return "none_box";
      case ir::TileLayout::row_major:
        return "row_major";
      case ir::TileLayout::col_major:
        return "col_major";
    }
    return "row_major";
  };

  std::ostringstream oss;
  oss << "!pto.tile_buf<loc=" << loc << ", dtype=" << dtype_str;
  oss << ", rows=" << rows << ", cols=" << cols;
  oss << ", v_row=" << rows << ", v_col=" << cols;
  oss << ", blayout=" << layout_to_str(blayout);
  oss << ", slayout=" << layout_to_str(slayout);
  oss << ", fractal=" << fractal;
  oss << ", pad=" << static_cast<int>(pad) << ">";
  return oss.str();
}

std::string PTOCodegen::GetExprTypeAnnotation(const ir::ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    // Check if this variable maps to a tile buffer via memref
    auto memref_it = var_to_memref_.find(var->name_);
    if (memref_it != var_to_memref_.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    // Check if this is a scalar parameter
    if (auto scalar_type = As<ScalarType>(var->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
    // Check if variable has TileType with memref
    if (auto tile_type = As<TileType>(var->GetType())) {
      if (tile_type->memref_.has_value()) {
        return GetTileBufTypeString(tile_type->memref_.value().get());
      }
    }
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return "f32";
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return "index";
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultTileBufTypeString() const {
  if (current_result_tile_type_ && current_result_tile_type_->memref_.has_value()) {
    return GetTileBufTypeString(current_result_tile_type_->memref_.value().get());
  }
  return "";
}

// ========================================================================
// Control flow helpers
// ========================================================================

std::string PTOCodegen::EmitArithBinaryOp(const std::string& mlir_op, const std::string& lhs,
                                          const std::string& rhs, const std::string& result_type) {
  std::string result = NewTemp();
  Emit(result + " = " + mlir_op + " " + lhs + ", " + rhs + " : " + result_type);
  return result;
}

std::string PTOCodegen::EmitArithCmpi(const std::string& predicate, const std::string& lhs,
                                      const std::string& rhs, const std::string& operand_type) {
  std::string result = NewTemp();
  Emit(result + " = arith.cmpi " + predicate + ", " + lhs + ", " + rhs + " : " + operand_type);
  return result;
}

void PTOCodegen::VisitBinaryArithExpr(const BinaryExprPtr& op, const std::string& int_op,
                                      const std::string& float_op) {
  VisitExpr(op->left_);
  std::string lhs = current_expr_value_;
  VisitExpr(op->right_);
  std::string rhs = current_expr_value_;

  // Determine type: use float op for float types, int op otherwise
  std::string result_type = "index";
  std::string mlir_op = int_op;
  if (auto scalar_type = As<ScalarType>(op->GetType())) {
    if (scalar_type->dtype_.IsFloat()) {
      result_type = GetTypeString(scalar_type->dtype_);
      mlir_op = float_op;
    }
  }
  current_expr_value_ = EmitArithBinaryOp(mlir_op, lhs, rhs, result_type);
}

void PTOCodegen::VisitCmpExpr(const BinaryExprPtr& op, const std::string& predicate) {
  VisitExpr(op->left_);
  std::string lhs = current_expr_value_;
  VisitExpr(op->right_);
  std::string rhs = current_expr_value_;

  // Determine operand type from the left operand
  std::string operand_type = "index";
  bool is_float = false;
  if (auto scalar_type = As<ScalarType>(op->left_->GetType())) {
    if (scalar_type->dtype_.IsFloat()) {
      operand_type = GetTypeString(scalar_type->dtype_);
      is_float = true;
    }
  }

  if (is_float) {
    static const std::map<std::string, std::string> pred_map = {
        {"eq", "oeq"}, {"ne", "one"}, {"slt", "olt"}, {"sle", "ole"}, {"sgt", "ogt"}, {"sge", "oge"}};
    auto it = pred_map.find(predicate);
    INTERNAL_CHECK(it != pred_map.end()) << "Unsupported float predicate for " << predicate;
    std::string float_pred = it->second;

    std::string result = NewTemp();
    Emit(result + " = arith.cmpf " + float_pred + ", " + lhs + ", " + rhs + " : " + operand_type);
    current_expr_value_ = result;
  } else {
    current_expr_value_ = EmitArithCmpi(predicate, lhs, rhs, operand_type);
  }
}

// ========================================================================
// Expression visitors - Leaf nodes
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::VarPtr& op) { current_expr_value_ = GetVarName(op); }

void PTOCodegen::VisitExpr_(const ir::IterArgPtr& op) {
  current_expr_value_ = GetVarName(std::dynamic_pointer_cast<const ir::Var>(op));
}

void PTOCodegen::VisitExpr_(const ir::ConstIntPtr& op) {
  current_expr_value_ = GetOrEmitIndexConstant(op->value_);
}

void PTOCodegen::VisitExpr_(const ir::ConstFloatPtr& op) {
  std::string mlir_type = "f32";
  if (auto scalar_type = As<ScalarType>(op->GetType())) {
    mlir_type = GetTypeString(scalar_type->dtype_);
  }
  current_expr_value_ = GetOrEmitFloatConstant(op->value_, mlir_type);
}

void PTOCodegen::VisitExpr_(const ir::ConstBoolPtr& op) {
  std::string result = NewTemp();
  std::string val = op->value_ ? "true" : "false";
  Emit(result + " = arith.constant " + val + " : i1");
  current_expr_value_ = result;
}

// ========================================================================
// Expression visitors - Binary arithmetic
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::AddPtr& op) { VisitBinaryArithExpr(op, "arith.addi", "arith.addf"); }
void PTOCodegen::VisitExpr_(const ir::SubPtr& op) { VisitBinaryArithExpr(op, "arith.subi", "arith.subf"); }
void PTOCodegen::VisitExpr_(const ir::MulPtr& op) { VisitBinaryArithExpr(op, "arith.muli", "arith.mulf"); }
void PTOCodegen::VisitExpr_(const ir::FloorDivPtr& op) {
  VisitBinaryArithExpr(op, "arith.divsi", "arith.divf");
}
void PTOCodegen::VisitExpr_(const ir::FloorModPtr& op) {
  VisitBinaryArithExpr(op, "arith.remsi", "arith.remf");
}

// ========================================================================
// Expression visitors - Comparisons
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::EqPtr& op) { VisitCmpExpr(op, "eq"); }
void PTOCodegen::VisitExpr_(const ir::NePtr& op) { VisitCmpExpr(op, "ne"); }
void PTOCodegen::VisitExpr_(const ir::LtPtr& op) { VisitCmpExpr(op, "slt"); }
void PTOCodegen::VisitExpr_(const ir::LePtr& op) { VisitCmpExpr(op, "sle"); }
void PTOCodegen::VisitExpr_(const ir::GtPtr& op) { VisitCmpExpr(op, "sgt"); }
void PTOCodegen::VisitExpr_(const ir::GePtr& op) { VisitCmpExpr(op, "sge"); }

// ========================================================================
// Statement visitors - Control flow
// ========================================================================

void PTOCodegen::VisitStmt_(const EvalStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";
  INTERNAL_CHECK(op->expr_ != nullptr) << "Internal error: EvalStmt has null expression";
  VisitExpr(op->expr_);
}

void PTOCodegen::VisitStmt_(const YieldStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null YieldStmt";

  if (op->value_.empty()) {
    return;
  }

  std::vector<std::string> yielded_values;
  for (const auto& expr : op->value_) {
    VisitExpr(expr);
    yielded_values.push_back(current_expr_value_);
    current_expr_value_ = "";
  }
  yield_buffer_ = yielded_values;
}

void PTOCodegen::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";
  INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: IfStmt has null condition";
  INTERNAL_CHECK(op->then_body_ != nullptr) << "Internal error: IfStmt has null then_body";

  // Evaluate condition
  VisitExpr(op->condition_);
  std::string condition = current_expr_value_;
  current_expr_value_ = "";

  if (op->return_vars_.empty()) {
    // Simple scf.if (no return values)
    Emit("scf.if " + condition + " {");
    indent_level_++;
    VisitStmt(op->then_body_);
    indent_level_--;

    if (op->else_body_.has_value()) {
      Emit("} else {");
      indent_level_++;
      VisitStmt(*op->else_body_);
      indent_level_--;
    }
    Emit("}");
  } else {
    // scf.if with return values
    std::vector<std::string> return_var_names;
    std::vector<std::string> return_var_types;
    for (const auto& return_var : op->return_vars_) {
      std::string ret_name = NewTemp();
      var_to_mlir_[return_var->name_] = ret_name;
      return_var_names.push_back(ret_name);
      // Default to index for scalar types
      std::string type_str = "index";
      if (auto scalar_type = As<ScalarType>(return_var->GetType())) {
        if (scalar_type->dtype_ == DataType::BOOL) {
          type_str = "i1";
        } else if (scalar_type->dtype_.IsFloat()) {
          type_str = GetTypeString(scalar_type->dtype_);
        }
      }
      return_var_types.push_back(type_str);
    }

    CHECK(op->else_body_.has_value()) << "IfStmt with return_vars requires else_body";

    // Emit: %ret0, %ret1 = scf.if %cond -> (type0, type1) {
    std::ostringstream oss;
    for (size_t i = 0; i < return_var_names.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << return_var_names[i];
    }
    oss << " = scf.if " << condition << " -> (";
    for (size_t i = 0; i < return_var_types.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << return_var_types[i];
    }
    oss << ") {";
    Emit(oss.str());
    indent_level_++;

    // Then branch
    yield_buffer_.clear();
    VisitStmt(op->then_body_);
    if (!yield_buffer_.empty()) {
      std::ostringstream yield_oss;
      yield_oss << "scf.yield ";
      for (size_t i = 0; i < yield_buffer_.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << yield_buffer_[i];
      }
      yield_oss << " : ";
      for (size_t i = 0; i < return_var_types.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << return_var_types[i];
      }
      Emit(yield_oss.str());
    }
    CHECK(yield_buffer_.size() == return_var_types.size())
        << "IfStmt then-branch yield count (" << yield_buffer_.size() << ") must match return_vars ("
        << return_var_types.size() << ")";
    yield_buffer_.clear();
    indent_level_--;

    // Else branch
    if (op->else_body_.has_value()) {
      Emit("} else {");
      indent_level_++;
      VisitStmt(*op->else_body_);
      if (!yield_buffer_.empty()) {
        std::ostringstream yield_oss;
        yield_oss << "scf.yield ";
        for (size_t i = 0; i < yield_buffer_.size(); ++i) {
          if (i > 0) yield_oss << ", ";
          yield_oss << yield_buffer_[i];
        }
        yield_oss << " : ";
        for (size_t i = 0; i < return_var_types.size(); ++i) {
          if (i > 0) yield_oss << ", ";
          yield_oss << return_var_types[i];
        }
        Emit(yield_oss.str());
      }
      CHECK(yield_buffer_.size() == return_var_types.size())
          << "IfStmt else-branch yield count (" << yield_buffer_.size() << ") must match return_vars ("
          << return_var_types.size() << ")";
      yield_buffer_.clear();
      indent_level_--;
    }
    Emit("}");
  }
}

void PTOCodegen::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";
  INTERNAL_CHECK(op->loop_var_ != nullptr) << "Internal error: ForStmt has null loop_var";
  INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: ForStmt has null body";

  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "ForStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  // Evaluate loop bounds
  VisitExpr(op->start_);
  std::string start = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->stop_);
  std::string stop = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->step_);
  std::string step = current_expr_value_;
  current_expr_value_ = "";

  // Register loop variable
  std::string loop_var_name = NewTemp();
  var_to_mlir_[op->loop_var_->name_] = loop_var_name;

  if (op->iter_args_.empty()) {
    // Simple scf.for (no iter_args)
    Emit("scf.for " + loop_var_name + " = " + start + " to " + stop + " step " + step + " {");
    indent_level_++;

    yield_buffer_.clear();
    VisitStmt(op->body_);

    indent_level_--;
    Emit("}");
  } else {
    // scf.for with iter_args
    std::vector<std::string> init_values;
    std::vector<std::string> iter_arg_names;
    std::vector<std::string> iter_arg_types;

    for (const auto& iter_arg : op->iter_args_) {
      VisitExpr(iter_arg->initValue_);
      init_values.push_back(current_expr_value_);
      current_expr_value_ = "";

      std::string iter_name = NewTemp();
      var_to_mlir_[iter_arg->name_] = iter_name;
      iter_arg_names.push_back(iter_name);

      std::string type_str = "index";
      if (auto scalar_type = As<ScalarType>(iter_arg->GetType())) {
        if (scalar_type->dtype_ == DataType::BOOL) {
          type_str = "i1";
        } else if (scalar_type->dtype_.IsFloat()) {
          type_str = GetTypeString(scalar_type->dtype_);
        }
      }
      iter_arg_types.push_back(type_str);
    }

    // Register return_vars SSA names
    std::vector<std::string> return_var_names;
    for (const auto& return_var : op->return_vars_) {
      std::string ret_name = NewTemp();
      var_to_mlir_[return_var->name_] = ret_name;
      return_var_names.push_back(ret_name);
    }

    // Emit: %ret0 = scf.for %i = %start to %stop step %step
    //           iter_args(%acc = %init) -> (type) {
    std::ostringstream oss;
    for (size_t i = 0; i < return_var_names.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << return_var_names[i];
    }
    oss << " = scf.for " << loop_var_name << " = " << start << " to " << stop << " step " << step;
    oss << " iter_args(";
    for (size_t i = 0; i < iter_arg_names.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << iter_arg_names[i] << " = " << init_values[i];
    }
    oss << ") -> (";
    for (size_t i = 0; i < iter_arg_types.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << iter_arg_types[i];
    }
    oss << ") {";
    Emit(oss.str());
    indent_level_++;

    yield_buffer_.clear();
    VisitStmt(op->body_);

    // Emit scf.yield from yield_buffer_
    if (!yield_buffer_.empty()) {
      std::ostringstream yield_oss;
      yield_oss << "scf.yield ";
      for (size_t i = 0; i < yield_buffer_.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << yield_buffer_[i];
      }
      yield_oss << " : ";
      for (size_t i = 0; i < iter_arg_types.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << iter_arg_types[i];
      }
      Emit(yield_oss.str());
    }
    CHECK(yield_buffer_.size() == iter_arg_types.size())
        << "ForStmt yield count (" << yield_buffer_.size() << ") must match iter_args ("
        << iter_arg_types.size() << ")";
    yield_buffer_.clear();

    indent_level_--;
    Emit("}");
  }
}

}  // namespace codegen
}  // namespace pypto
