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

#include <cctype>
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
#include "pypto/ir/transforms/utils/memref_utils.h"
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
using ir::WhileStmtPtr;
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
  } else if (dtype == ::pypto::DataType::INDEX) {
    return "index";
  } else if (dtype == ::pypto::DataType::INT64) {
    return "i64";
  } else if (dtype == ::pypto::DataType::INT8) {
    return "i8";
  } else if (dtype == ::pypto::DataType::UINT8) {
    return "ui8";
  } else if (dtype == ::pypto::DataType::BOOL) {
    return "i1";
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
  } else if (space == ir::MemorySpace::Bias) {
    return "bias";
  } else {
    throw pypto::ValueError("Invalid MemorySpace value");
  }
}

/// Join a vector of strings with ", " separator
static std::string JoinCommaSep(const std::vector<std::string>& items) {
  std::ostringstream oss;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << items[i];
  }
  return oss.str();
}

/// Join pairs of strings as "a sep b" with ", " between pairs
static std::string JoinPairs(const std::vector<std::string>& lhs, const std::string& sep,
                             const std::vector<std::string>& rhs) {
  INTERNAL_CHECK(lhs.size() == rhs.size()) << "Internal error: JoinPairs size mismatch";
  std::ostringstream oss;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << lhs[i] << sep << rhs[i];
  }
  return oss.str();
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
    if (iter_arg_ids_.count(op->UniqueId())) return;
    if (auto tile_type = ir::GetTileTypeWithMemRef(op->GetType())) {
      AddMemRefIfUnique(ir::GetDefinedMemRef(tile_type), tile_type);
    }
  }

  void VisitExpr_(const ir::IterArgPtr& op) override {
    iter_arg_ids_.insert(op->UniqueId());
    ir::IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const ir::MemRef*> seen_ptrs_;
  std::map<const ir::MemRef*, std::shared_ptr<const TileType>> memref_tile_types_;
  std::set<uint64_t> iter_arg_ids_;

  void AddMemRefIfUnique(const MemRefPtr& memref, const std::shared_ptr<const TileType>& tile_type) {
    const ir::MemRef* raw_ptr = memref.get();
    if (ir::TryRegisterUniqueMemRef(memref, seen_ptrs_)) {
      memrefs_.push_back(memref);
      memref_tile_types_[raw_ptr] = tile_type;
    } else {
      // Merge TileView properties when multiple tiles share the same MemRef:
      // - Keep valid_shape from the original tile (e.g., from load)
      // - Take pad from the new tile if it has a non-null pad (e.g., from fillpad)
      // This ensures fillpad's pad_value is used while preserving the original valid_shape
      auto existing = memref_tile_types_[raw_ptr];
      if (tile_type->tile_view_.has_value() && tile_type->tile_view_->pad != ir::PadValue::null) {
        // Merge: keep valid_shape from existing, take pad from new tile
        ir::TileView merged_view;
        if (existing->tile_view_.has_value()) {
          merged_view = existing->tile_view_.value();
        }
        merged_view.pad = tile_type->tile_view_->pad;
        auto merged_tile_type = std::make_shared<TileType>(
            existing->shape_, existing->dtype_, existing->memref_, merged_view, existing->memory_space_);
        memref_tile_types_[raw_ptr] = merged_tile_type;
      }
    }
  }
};

// ========================================================================
// Constructors
// ========================================================================

PTOCodegen::PTOCodegen() : backend_(backend::GetBackend()) {
  auto type = backend::GetBackendType();
  CHECK(type == backend::BackendType::Ascend910B_PTO || type == backend::BackendType::Ascend950)
      << "PTOCodegen requires Ascend910B_PTO or Ascend950 backend, but "
      << (type == backend::BackendType::Ascend910B_CCE ? "Ascend910B_CCE" : "unknown") << " is configured";
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
    INTERNAL_CHECK(ir::IsInCoreType(func->func_type_))
        << "PTO backend only supports InCore-variant functions (InCore, AIC, AIV), but function '"
        << func->name_ << "' has type " << ir::FunctionTypeToString(func->func_type_);
    GenerateFunction(func);
  }

  stream_ << "}\n";
  return stream_.str();
}

void PTOCodegen::GenerateFunction(const FunctionPtr& func) {
  current_function_ = func;
  current_result_var_.reset();
  current_result_buf_.clear();
  current_result_tile_type_ = nullptr;
  temp_counter_ = 0;
  used_ssa_names_.clear();
  memref_to_var_name_.clear();
  var_to_mlir_.clear();
  tensor_to_view_.clear();
  memref_to_mlir_.clear();
  var_to_memref_.clear();
  memref_to_tile_type_.clear();
  emitted_constants_.clear();
  emitted_float_constants_.clear();
  float_const_names_.clear();
  extra_alloc_tiles_.clear();
  extra_tile_buf_types_.clear();
  constants_section_.str("");
  constants_section_.clear();
  body_section_.str("");
  body_section_.clear();

  // Reserve %argN names upfront so NewNamedTemp never collides with them
  for (size_t i = 0; i < func->params_.size(); i++) {
    used_ssa_names_.insert("arg" + std::to_string(i));
  }
  // Also reserve extra %argN for dynamic dimension parameters
  {
    size_t extra = 0;
    for (const auto& param : func->params_) {
      if (auto tensor_type = As<TensorType>(param->GetType())) {
        std::set<const ir::Var*> seen;
        for (const auto& dim : tensor_type->shape_) {
          if (auto var = As<ir::Var>(dim)) {
            if (seen.insert(GetVarKey(var)).second) {
              extra++;
            }
          }
        }
      }
    }
    for (size_t i = 0; i < extra; i++) {
      used_ssa_names_.insert("arg" + std::to_string(func->params_.size() + i));
    }
  }

  BuildVarToMemRefMapping(func);

  MemRefCollectorVisitor collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }

  for (const auto& memref : collector.GetMemRefs()) {
    auto name_it = memref_to_var_name_.find(memref.get());
    std::string tile_buf = (name_it != memref_to_var_name_.end()) ? NewNamedTemp(name_it->second) : NewTemp();
    memref_to_mlir_[memref.get()] = tile_buf;
  }
  memref_to_tile_type_ = collector.GetMemRefTileTypes();

  // Collect ordered unique dynamic dimension variables from tensor parameter shapes
  std::vector<VarPtr> dyn_vars;
  {
    std::set<const ir::Var*> seen_dyn_vars;
    for (const auto& param : func->params_) {
      if (auto tensor_type = As<TensorType>(param->GetType())) {
        for (const auto& dim : tensor_type->shape_) {
          if (auto var = As<ir::Var>(dim)) {
            if (seen_dyn_vars.insert(GetVarKey(var)).second) {
              dyn_vars.push_back(var);
            }
          }
        }
      }
    }
  }

  stream_ << "  func.func @" << func->name_ << "(";

  std::set<const ir::Var*> param_keys;
  for (size_t i = 0; i < func->params_.size(); i++) {
    if (i > 0) stream_ << ", ";
    const auto& param = func->params_[i];
    std::string arg_name = "%arg" + std::to_string(i);
    stream_ << arg_name << ": ";

    auto param_key = GetVarKey(param);
    BindVarToMlir(param, arg_name);
    param_keys.insert(param_key);

    if (auto tensor_type = As<TensorType>(param->GetType())) {
      stream_ << "!pto.ptr<" << GetTypeString(tensor_type->dtype_) << ">";
    } else if (auto scalar_type = As<ScalarType>(param->GetType())) {
      stream_ << GetTypeString(scalar_type->dtype_);
    } else {
      stream_ << "!pto.ptr<f32>";
    }
  }

  // Append trailing index parameters for each unique dynamic dimension variable
  size_t next_arg_idx = func->params_.size();
  for (const auto& dyn_var : dyn_vars) {
    std::string arg_name = "%arg" + std::to_string(next_arg_idx++);
    stream_ << ", " << arg_name << ": index";
    BindVarToMlir(dyn_var, arg_name);
  }

  stream_ << ") {\n";
  indent_level_++;

  for (const auto& [var_key, memref_ptr] : var_to_memref_) {
    if (param_keys.find(var_key) == param_keys.end()) {
      var_to_mlir_[var_key] = memref_to_mlir_[memref_ptr];
    }
  }

  for (const auto& var : func->params_) {
    if (auto tensor_type = As<TensorType>(var->GetType())) {
      std::string tensor_view = NewNamedTemp(var->name_hint_ + "_view");
      BindTensorView(var, tensor_view);

      for (const auto& j : tensor_type->shape_) {
        if (As<ir::ConstInt>(j)) {
          GetOrEmitIndexConstant(GetConstIntValue(j));
        }
      }
      if (tensor_type->shape_.size() == 2) {
        if (As<ir::ConstInt>(tensor_type->shape_[1])) {
          GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[1]));
        }
        GetOrEmitIndexConstant(1);
      } else {
        // 1-D and N-D (N>2): pre-emit constant 1 (innermost stride). For N>2,
        // other strides are computed dynamically via arith.muli in
        // EmitMakeTensorViews to support dynamic dims.
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
  EmitExtraAllocTiles();
  stream_ << body_content;
  stream_ << GetIndent() << "return\n";

  indent_level_--;
  stream_ << "  }\n";
}

void PTOCodegen::BuildVarToMemRefMapping(const FunctionPtr& func) {
  class VarMemRefMapper : public ir::IRVisitor {
   public:
    std::map<const ir::Var*, const ir::MemRef*>& var_to_memref;
    std::map<const ir::MemRef*, std::string>& memref_to_var_name;

    VarMemRefMapper(std::map<const ir::Var*, const ir::MemRef*>& mapping,
                    std::map<const ir::MemRef*, std::string>& reverse_mapping)
        : var_to_memref(mapping), memref_to_var_name(reverse_mapping) {}

    void VisitStmt_(const AssignStmtPtr& op) override {
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        const auto memref = ir::GetDefinedMemRef(tile_type);
        const ir::MemRef* ptr = memref.get();
        var_to_memref[op->var_.get()] = ptr;
        // Record first variable name per MemRef (program order)
        if (memref_to_var_name.find(ptr) == memref_to_var_name.end()) {
          memref_to_var_name[ptr] = op->var_->name_hint_;
        }
      }
      ir::IRVisitor::VisitStmt_(op);
    }
  };

  VarMemRefMapper mapper(var_to_memref_, memref_to_var_name_);
  if (func->body_) {
    mapper.VisitStmt(func->body_);
  }
}

void PTOCodegen::EmitMakeTensorViews(const FunctionPtr& func) {
  for (size_t i = 0; i < func->params_.size(); i++) {
    const auto& param = func->params_[i];
    if (auto tensor_type = As<TensorType>(param->GetType())) {
      std::string tensor_view = tensor_to_view_.at(GetVarKey(param));

      bool layout_DN = false;
      if (tensor_type->tensor_view_.has_value()) {
        if (tensor_type->tensor_view_.value().layout == ir::TensorLayout::DN) {
          layout_DN = true;
        }
      }

      // For N-D (N > 2): pre-compute row-major strides as SSA values using arith.muli
      // so that dynamic dimensions (ir::Var) are handled correctly. Emit any needed
      // multiply instructions BEFORE the make_tensor_view line.
      std::vector<std::string> nd_stride_names;
      if (tensor_type->shape_.size() > 2) {
        const size_t rank = tensor_type->shape_.size();
        nd_stride_names.resize(rank);
        nd_stride_names[rank - 1] = GetOrEmitIndexConstant(1);
        for (int j = static_cast<int>(rank) - 2; j >= 0; j--) {
          std::string dim_mlir;
          if (auto var = As<ir::Var>(tensor_type->shape_[j + 1])) {
            dim_mlir = GetVarName(var);
          } else {
            dim_mlir = GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[j + 1]));
          }
          std::string mul_name = NewNamedTemp(param->name_hint_ + "_s" + std::to_string(j));
          stream_ << GetIndent() << mul_name << " = arith.muli " << nd_stride_names[j + 1] << ", " << dim_mlir
                  << " : index\n";
          nd_stride_names[j] = mul_name;
        }
      }

      stream_ << GetIndent() << tensor_view << " = pto.make_tensor_view ";
      stream_ << "%arg" << i;

      stream_ << ", shape = [";
      for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
        if (j > 0) stream_ << ", ";
        if (auto var = As<ir::Var>(tensor_type->shape_[j])) {
          stream_ << GetVarName(var);
        } else {
          stream_ << GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[j]));
        }
      }
      stream_ << "],";

      stream_ << " strides = [";
      if (tensor_type->shape_.size() == 2) {
        std::string row_stride;
        int idx = layout_DN ? 0 : 1;
        if (auto var = As<ir::Var>(tensor_type->shape_[idx])) {
          row_stride = GetVarName(var);
        } else {
          row_stride = GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[idx]));
        }
        if (layout_DN) {
          stream_ << GetOrEmitIndexConstant(1) << ", " << row_stride;
        } else {
          stream_ << row_stride << ", " << GetOrEmitIndexConstant(1);
        }
      } else if (tensor_type->shape_.size() == 1) {
        stream_ << GetOrEmitIndexConstant(1);
      } else {
        // Use pre-computed SSA stride names (built above via arith.muli)
        for (size_t j = 0; j < nd_stride_names.size(); j++) {
          if (j > 0) stream_ << ", ";
          stream_ << nd_stride_names[j];
        }
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

    // Collect dynamic valid_shape variable names if present
    std::string valid_row_mlir;
    std::string valid_col_mlir;
    auto tile_it = memref_to_tile_type_.find(memref.get());
    if (tile_it != memref_to_tile_type_.end()) {
      const auto& tile_type = tile_it->second;
      if (tile_type->tile_view_.has_value()) {
        const auto& tv = tile_type->tile_view_.value();
        if (tv.valid_shape.size() >= 1) {
          if (auto var = As<ir::Var>(tv.valid_shape[0])) {
            valid_row_mlir = GetVarName(var);
          }
        }
        if (tv.valid_shape.size() >= 2) {
          if (auto var = As<ir::Var>(tv.valid_shape[1])) {
            valid_col_mlir = GetVarName(var);
          }
        }
      }
    }

    std::ostringstream line;
    line << tile_buf << " = pto.alloc_tile";
    if (!valid_row_mlir.empty()) line << " valid_row = " << valid_row_mlir;
    if (!valid_col_mlir.empty()) line << " valid_col = " << valid_col_mlir;
    line << " : " << GetTileBufTypeString(memref.get());
    stream_ << GetIndent() << line.str() << "\n";
  }
}

// ========================================================================
// Private helpers
// ========================================================================

std::string PTOCodegen::GetIndent() const { return std::string(static_cast<size_t>(indent_level_) * 2, ' '); }

std::string PTOCodegen::GetOrEmitIndexConstant(int64_t value) {
  auto it = emitted_constants_.find(value);
  if (it != emitted_constants_.end()) {
    return it->second;
  }
  std::string ssa_id = "c" + std::to_string(value);
  std::string name;
  if (used_ssa_names_.find(ssa_id) == used_ssa_names_.end()) {
    used_ssa_names_.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }
  constants_section_ << GetIndent() << name << " = arith.constant " << value << " : index\n";
  emitted_constants_[value] = name;
  return name;
}

std::string PTOCodegen::GetTileBufForMemRef(const MemRefPtr& memref) {
  auto it = memref_to_mlir_.find(memref.get());
  INTERNAL_CHECK(it != memref_to_mlir_.end()) << "MemRef not found in mapping";
  return it->second;
}

std::string PTOCodegen::AllocNewTileBuf(const std::string& tile_buf_type_string,
                                        const std::string& name_hint) {
  std::string name = NewNamedTemp(name_hint);
  extra_alloc_tiles_.emplace_back(name, tile_buf_type_string);
  extra_tile_buf_types_[name] = tile_buf_type_string;
  return name;
}

void PTOCodegen::SetCurrentResultBuf(const std::string& buf) { current_result_buf_ = buf; }

void PTOCodegen::RegisterTileBufType(const std::string& ssa_name, const std::string& type_string) {
  extra_tile_buf_types_[ssa_name] = type_string;
}

void PTOCodegen::EmitExtraAllocTiles() {
  for (const auto& [name, type_str] : extra_alloc_tiles_) {
    stream_ << GetIndent() << name << " = pto.alloc_tile : " << type_str << "\n";
  }
}

// ========================================================================
// Statement visitors
// ========================================================================

void PTOCodegen::VisitStmt_(const AssignStmtPtr& op) {
  if (auto call = As<ir::Call>(op->value_)) {
    if (backend_ != nullptr && backend_->GetOpInfo(call->op_->name_) != nullptr) {
      std::string result_buf =
          op->var_->name_hint_;  // Seed for readable MLIR names when no tile buffer exists.
      std::shared_ptr<const TileType> result_tile_type;
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        result_buf = GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
        result_tile_type = tile_type;
      } else if (auto tile_type = As<TileType>(op->var_->GetType())) {
        result_tile_type = tile_type;
      } else if (As<ScalarType>(op->var_->GetType())) {
        // Pre-allocate an SSA name for scalar-result backend ops (e.g., tile.getval).
        // Register it in var_to_mlir_ so subsequent expressions can resolve the variable.
        result_buf = NewNamedTemp(op->var_->name_hint_);
        BindVarToMlir(op->var_, result_buf);
      }
      current_result_var_ = op->var_;
      current_result_buf_ = result_buf;
      current_result_tile_type_ = result_tile_type;
      VisitExpr(op->value_);
      // If codegen changed the result buffer (e.g., reshape allocated a new tile),
      // update variable mapping so subsequent references use the new buffer
      if (!current_result_buf_.empty() && current_result_buf_ != result_buf) {
        BindVarToMlir(op->var_, current_result_buf_);
      }
      current_result_var_.reset();
      current_result_buf_.clear();
      current_result_tile_type_ = nullptr;
      return;
    }
  }

  current_expr_value_ = "";
  VisitExpr(op->value_);
  // Register scalar/index result so subsequent expressions can look up this variable
  if (As<ScalarType>(op->var_->GetType()) && !current_expr_value_.empty()) {
    BindVarToMlir(op->var_, current_expr_value_);
  }
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

ir::VarPtr PTOCodegen::GetCurrentResultVar() const { return current_result_var_; }

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

const ir::Var* PTOCodegen::GetVarKey(const VarPtr& var) const {
  INTERNAL_CHECK(var != nullptr) << "Internal error: variable key requested for null Var";
  return var.get();
}

void PTOCodegen::BindVarToMlir(const VarPtr& var, const std::string& mlir_name) {
  var_to_mlir_[GetVarKey(var)] = mlir_name;
}

void PTOCodegen::BindTensorView(const VarPtr& var, const std::string& tensor_view_name) {
  tensor_to_view_[GetVarKey(var)] = tensor_view_name;
}

void PTOCodegen::BindVarToMemRef(const VarPtr& var, const ir::MemRef* memref) {
  var_to_memref_[GetVarKey(var)] = memref;
}

std::string PTOCodegen::GetVarName(const VarPtr& var) {
  auto key = GetVarKey(var);
  auto it = var_to_mlir_.find(key);
  if (it != var_to_mlir_.end()) {
    return it->second;
  }
  auto memref_it = var_to_memref_.find(key);
  if (memref_it != var_to_memref_.end()) {
    auto mlir_it = memref_to_mlir_.find(memref_it->second);
    if (mlir_it != memref_to_mlir_.end()) {
      return mlir_it->second;
    }
  }
  if (auto tile_type = ir::GetTileTypeWithMemRef(var->GetType())) {
    return GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
  }
  LOG_ERROR << "Variable " << var->name_hint_ << " not found in MLIR mapping";
  return "";
}

std::string PTOCodegen::NewTemp() {
  std::string name = std::to_string(temp_counter_++);
  while (used_ssa_names_.count(name)) {
    name = std::to_string(temp_counter_++);
  }
  used_ssa_names_.insert(name);
  return "%" + name;
}

std::string PTOCodegen::NewNamedTemp(const std::string& name) {
  // Sanitize name to be a valid MLIR SSA identifier: [a-zA-Z_][a-zA-Z0-9_$.]*
  std::string sanitized = name;
  if (!sanitized.empty()) {
    for (auto& c : sanitized) {
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '.' && c != '$') {
        c = '_';
      }
    }
    if (std::isdigit(static_cast<unsigned char>(sanitized[0]))) {
      sanitized.insert(0, 1, '_');
    }
  }

  if (!sanitized.empty() && used_ssa_names_.find(sanitized) == used_ssa_names_.end()) {
    used_ssa_names_.insert(sanitized);
    return "%" + sanitized;
  }
  return NewTemp();
}

void PTOCodegen::RegisterVarToMlir(const VarPtr& var, const std::string& mlir_name) {
  BindVarToMlir(var, mlir_name);
}

void PTOCodegen::RegisterTensorView(const VarPtr& var, const std::string& tensor_view_name) {
  BindTensorView(var, tensor_view_name);
}

int64_t PTOCodegen::GetConstIntValue(const ExprPtr& expr) {
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return const_int->value_;
  }
  LOG_ERROR << "Expected ConstInt expression";
  return 0;
}

std::string PTOCodegen::GetOrCreateTensorView(const VarPtr& tensor_var) {
  auto it = tensor_to_view_.find(GetVarKey(tensor_var));
  if (it != tensor_to_view_.end()) return it->second;
  // For IterArg, follow initValue_ chain to the original tensor parameter
  if (auto iter_arg = As<ir::IterArg>(tensor_var)) {
    if (auto init_var = As<ir::Var>(iter_arg->initValue_)) {
      return GetOrCreateTensorView(init_var);
    }
    if (auto init_iter = As<ir::IterArg>(iter_arg->initValue_)) {
      return GetOrCreateTensorView(init_iter);
    }
  }
  INTERNAL_CHECK(false) << "Tensor view not found for parameter: " << tensor_var->name_hint_;
  return "";
}

std::string PTOCodegen::GetIndexConstant(int64_t val) { return GetOrEmitIndexConstant(val); }

std::string PTOCodegen::GetOrEmitFloatConstant(double value, const std::string& mlir_type) {
  if (emitted_float_constants_.find(value) == emitted_float_constants_.end()) {
    std::string ssa_id = "cst";
    if (!emitted_float_constants_.empty()) {
      ssa_id += "_" + std::to_string(emitted_float_constants_.size());
    }
    std::string name;
    if (used_ssa_names_.find(ssa_id) == used_ssa_names_.end()) {
      used_ssa_names_.insert(ssa_id);
      name = "%" + ssa_id;
    } else {
      name = NewTemp();
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

// Helper to convert TileLayout to string
static const char* TileLayoutToStr(ir::TileLayout layout) {
  switch (layout) {
    case ir::TileLayout::none_box:
      return "none_box";
    case ir::TileLayout::row_major:
      return "row_major";
    case ir::TileLayout::col_major:
      return "col_major";
    default:
      INTERNAL_CHECK(false) << "Unknown TileLayout: " << static_cast<int>(layout);
      return "";  // Should be unreachable
  }
}

// Helper to format tile_buf type string from components
// v_row/v_col are the valid shape dimensions (may differ from rows/cols when valid_shapes is specified)
static std::string FormatTileBufTypeString(const std::string& loc, const std::string& dtype_str, int64_t rows,
                                           int64_t cols, ir::TileLayout blayout, ir::TileLayout slayout,
                                           uint64_t fractal, ir::PadValue pad, int64_t v_row, int64_t v_col,
                                           bool v_row_dynamic = false, bool v_col_dynamic = false) {
  std::ostringstream oss;
  oss << "!pto.tile_buf<loc=" << loc << ", dtype=" << dtype_str;
  oss << ", rows=" << rows << ", cols=" << cols;
  oss << ", v_row=" << (v_row_dynamic ? "?" : std::to_string(v_row));
  oss << ", v_col=" << (v_col_dynamic ? "?" : std::to_string(v_col));
  oss << ", blayout=" << TileLayoutToStr(blayout);
  oss << ", slayout=" << TileLayoutToStr(slayout);
  oss << ", fractal=" << fractal;
  oss << ", pad=" << static_cast<int>(pad) << ">";
  return oss.str();
}

// Extract dtype, shape and layout from a TileType into output parameters
// v_row/v_col are set to valid_shape values if available, otherwise to rows/cols
static void ExtractTileTypeInfo(const TileType& tile_type, const PTOCodegen& codegen, std::string& dtype_str,
                                int64_t& rows, int64_t& cols, ir::TileLayout& blayout,
                                ir::TileLayout& slayout, uint64_t& fractal, ir::PadValue& pad, int64_t& v_row,
                                int64_t& v_col, bool& v_row_dynamic, bool& v_col_dynamic) {
  dtype_str = codegen.GetTypeString(tile_type.dtype_);
  if (tile_type.shape_.size() >= 2) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) rows = c0->value_;
    if (auto c1 = As<ir::ConstInt>(tile_type.shape_[1])) cols = c1->value_;
  } else if (tile_type.shape_.size() == 1) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) {
      rows = 1;
      cols = c0->value_;
    }
  }
  // Default v_row/v_col to physical shape
  v_row = rows;
  v_col = cols;
  if (tile_type.tile_view_.has_value()) {
    const auto& tv = *tile_type.tile_view_;
    blayout = tv.blayout;
    slayout = tv.slayout;
    fractal = tv.fractal;
    pad = tv.pad;
    // Extract valid_shape values
    if (tv.valid_shape.size() >= 1) {
      if (auto c0 = As<ir::ConstInt>(tv.valid_shape[0])) {
        v_row = c0->value_;
      } else if (As<ir::Var>(tv.valid_shape[0])) {
        v_row_dynamic = true;
      }
    }
    if (tv.valid_shape.size() >= 2) {
      if (auto c1 = As<ir::ConstInt>(tv.valid_shape[1])) {
        v_col = c1->value_;
      } else if (As<ir::Var>(tv.valid_shape[1])) {
        v_col_dynamic = true;
      }
    }
  } else if (cols == 1 && rows > 1) {
    // Infer blayout from shape: column vectors [N, 1] use col_major (DN format convention)
    blayout = ir::TileLayout::col_major;
  }
}

std::string PTOCodegen::GetTileBufTypeString(const ir::MemRef* memref) const {
  auto tile_it = memref_to_tile_type_.find(memref);
  INTERNAL_CHECK(tile_it != memref_to_tile_type_.end())
      << "Internal error: missing tile type for MemRef '" << memref->name_hint_ << "'";
  auto memory_space = tile_it->second->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value()) << "Internal error: tile type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::PadValue pad = ir::PadValue::null;

  int64_t v_row = rows;
  int64_t v_col = cols;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;
  ExtractTileTypeInfo(*tile_it->second, *this, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row,
                      v_col, v_row_dynamic, v_col_dynamic);

  return FormatTileBufTypeString(loc, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row, v_col,
                                 v_row_dynamic, v_col_dynamic);
}

std::string PTOCodegen::GetTileBufTypeStringFromTileType(
    const std::shared_ptr<const ir::TileType>& tile_type) const {
  INTERNAL_CHECK(tile_type) << "Internal error: tile_type must not be null";
  auto memory_space = tile_type->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value()) << "Internal error: tile_type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::PadValue pad = ir::PadValue::null;
  int64_t v_row = rows;
  int64_t v_col = cols;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;

  ExtractTileTypeInfo(*tile_type, *this, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row, v_col,
                      v_row_dynamic, v_col_dynamic);

  return FormatTileBufTypeString(loc, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row, v_col,
                                 v_row_dynamic, v_col_dynamic);
}

std::string PTOCodegen::GetExprTypeAnnotation(const ir::ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    auto key = GetVarKey(var);
    // Check if variable was remapped to a dynamically-allocated tile buffer (e.g., reshape output)
    auto mlir_it = var_to_mlir_.find(key);
    if (mlir_it != var_to_mlir_.end()) {
      auto extra_it = extra_tile_buf_types_.find(mlir_it->second);
      if (extra_it != extra_tile_buf_types_.end()) {
        return extra_it->second;
      }
    }
    // Check if this variable maps to a tile buffer via memref
    auto memref_it = var_to_memref_.find(key);
    if (memref_it != var_to_memref_.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    // Check if this is a scalar parameter
    if (auto scalar_type = As<ScalarType>(var->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
    // Check if variable has TileType with memref
    if (auto tile_type = ir::GetTileTypeWithMemRef(var->GetType())) {
      return GetTileBufTypeString(ir::GetDefinedMemRef(tile_type).get());
    }
  }
  if (auto iter_arg = As<ir::IterArg>(expr)) {
    auto memref_it = var_to_memref_.find(GetVarKey(std::dynamic_pointer_cast<const ir::Var>(iter_arg)));
    if (memref_it != var_to_memref_.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      return GetTileBufTypeString(ir::GetDefinedMemRef(tile_type).get());
    }
    if (auto scalar_type = As<ScalarType>(iter_arg->GetType())) {
      return GetTypeString(scalar_type->dtype_);
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

std::string PTOCodegen::GetCurrentResultTileBufTypeStringFromTileType() const {
  if (current_result_tile_type_ && current_result_tile_type_->memref_.has_value()) {
    return GetTileBufTypeStringFromTileType(current_result_tile_type_);
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

  // Determine type: float op for float types, exact integer type otherwise
  std::string result_type = "index";
  std::string mlir_op = int_op;
  if (auto scalar_type = As<ScalarType>(op->GetType())) {
    if (scalar_type->dtype_.IsFloat()) {
      result_type = GetTypeString(scalar_type->dtype_);
      mlir_op = float_op;
    } else if (scalar_type->dtype_ != ::pypto::DataType::INDEX) {
      result_type = GetTypeString(scalar_type->dtype_);
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
    } else if (scalar_type->dtype_ != ::pypto::DataType::INDEX) {
      operand_type = GetTypeString(scalar_type->dtype_);
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
  std::string val = op->value_ ? "1" : "0";
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

void PTOCodegen::VisitExpr_(const ir::CastPtr& op) {
  VisitExpr(op->operand_);
  std::string src = current_expr_value_;

  ::pypto::DataType src_dtype = ir::GetScalarDtype(op->operand_);
  ::pypto::DataType dst_dtype = ir::GetScalarDtype(op);
  std::string src_type = GetTypeString(src_dtype);
  std::string dst_type = GetTypeString(dst_dtype);

  std::string result = NewTemp();
  bool src_is_index = (src_dtype == ::pypto::DataType::INDEX);
  bool dst_is_index = (dst_dtype == ::pypto::DataType::INDEX);
  bool src_is_float = src_dtype.IsFloat();
  bool dst_is_float = dst_dtype.IsFloat();

  bool src_is_uint = src_dtype.IsUnsignedInt();
  bool dst_is_uint = dst_dtype.IsUnsignedInt();

  std::string mlir_op;
  if (src_dtype == dst_dtype) {
    // No-op: same type (includes index → index)
    current_expr_value_ = src;
    return;
  } else if (src_is_index || dst_is_index) {
    // index <-> integer only; float <-> index is not valid in MLIR
    CHECK(!src_is_float && !dst_is_float) << "Cast between float and index types is not supported";
    mlir_op = "arith.index_cast";
  } else if (src_is_float && dst_is_float) {
    mlir_op = (dst_dtype.GetBit() > src_dtype.GetBit()) ? "arith.extf" : "arith.truncf";
  } else if (!src_is_float && !dst_is_float) {
    if (dst_dtype.GetBit() > src_dtype.GetBit()) {
      mlir_op = src_is_uint ? "arith.extui" : "arith.extsi";
    } else {
      mlir_op = "arith.trunci";
    }
  } else if (!src_is_float && dst_is_float) {
    mlir_op = src_is_uint ? "arith.uitofp" : "arith.sitofp";
  } else {
    mlir_op = dst_is_uint ? "arith.fptoui" : "arith.fptosi";
  }

  Emit(result + " = " + mlir_op + " " + src + " : " + src_type + " to " + dst_type);
  current_expr_value_ = result;
}

// ========================================================================
// Expression visitors - Logical & Bitwise
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::AndPtr& op) { VisitBinaryArithExpr(op, "arith.andi", "arith.andi"); }
void PTOCodegen::VisitExpr_(const ir::OrPtr& op) { VisitBinaryArithExpr(op, "arith.ori", "arith.ori"); }
void PTOCodegen::VisitExpr_(const ir::XorPtr& op) { VisitBinaryArithExpr(op, "arith.xori", "arith.xori"); }
void PTOCodegen::VisitExpr_(const ir::BitAndPtr& op) { VisitBinaryArithExpr(op, "arith.andi", "arith.andi"); }
void PTOCodegen::VisitExpr_(const ir::BitOrPtr& op) { VisitBinaryArithExpr(op, "arith.ori", "arith.ori"); }
void PTOCodegen::VisitExpr_(const ir::BitXorPtr& op) { VisitBinaryArithExpr(op, "arith.xori", "arith.xori"); }
void PTOCodegen::VisitExpr_(const ir::BitShiftLeftPtr& op) {
  VisitBinaryArithExpr(op, "arith.shli", "arith.shli");
}
void PTOCodegen::VisitExpr_(const ir::BitShiftRightPtr& op) {
  // Use unsigned shift (shrui) for unsigned integer types, signed shift (shrsi) otherwise
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.shrui" : "arith.shrsi";
  VisitBinaryArithExpr(op, int_op, int_op);
}

// ========================================================================
// Expression visitors - Other binary
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::FloatDivPtr& op) {
  // Use unsigned division (divui) for unsigned integer types, signed division (divsi) otherwise
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.divui" : "arith.divsi";
  VisitBinaryArithExpr(op, int_op, "arith.divf");
}
void PTOCodegen::VisitExpr_(const ir::MinPtr& op) {
  // Use unsigned min (minui) for unsigned integer types, signed min (minsi) otherwise
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.minui" : "arith.minsi";
  VisitBinaryArithExpr(op, int_op, "arith.minimumf");
}
void PTOCodegen::VisitExpr_(const ir::MaxPtr& op) {
  // Use unsigned max (maxui) for unsigned integer types, signed max (maxsi) otherwise
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string int_op = dtype.IsUnsignedInt() ? "arith.maxui" : "arith.maxsi";
  VisitBinaryArithExpr(op, int_op, "arith.maximumf");
}

// ========================================================================
// Expression visitors - Unary
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::NotPtr& op) {
  VisitExpr(op->operand_);
  std::string src = current_expr_value_;
  ::pypto::DataType src_dtype = ir::GetScalarDtype(op->operand_);
  std::string src_type = GetTypeString(src_dtype);
  std::string zero = NewTemp();
  std::string result = NewTemp();
  if (src_dtype.IsFloat()) {
    Emit(zero + " = arith.constant 0.0 : " + src_type);
    Emit(result + " = arith.cmpf oeq, " + src + ", " + zero + " : " + src_type);
  } else {
    Emit(zero + " = arith.constant 0 : " + src_type);
    Emit(result + " = arith.cmpi eq, " + src + ", " + zero + " : " + src_type);
  }
  current_expr_value_ = result;
}

void PTOCodegen::VisitExpr_(const ir::NegPtr& op) {
  VisitExpr(op->operand_);
  std::string src = current_expr_value_;
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string type_str = GetTypeString(dtype);
  std::string result = NewTemp();
  if (dtype.IsFloat()) {
    Emit(result + " = arith.negf " + src + " : " + type_str);
  } else {
    std::string zero = NewTemp();
    Emit(zero + " = arith.constant 0 : " + type_str);
    Emit(result + " = arith.subi " + zero + ", " + src + " : " + type_str);
  }
  current_expr_value_ = result;
}

void PTOCodegen::VisitExpr_(const ir::AbsPtr& op) {
  VisitExpr(op->operand_);
  std::string src = current_expr_value_;
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string type_str = GetTypeString(dtype);
  std::string result = NewTemp();
  if (dtype.IsFloat()) {
    Emit(result + " = math.absf " + src + " : " + type_str);
  } else {
    Emit(result + " = math.absi " + src + " : " + type_str);
  }
  current_expr_value_ = result;
}

void PTOCodegen::VisitExpr_(const ir::BitNotPtr& op) {
  VisitExpr(op->operand_);
  std::string src = current_expr_value_;
  ::pypto::DataType dtype = ir::GetScalarDtype(op);
  std::string type_str = GetTypeString(dtype);
  std::string all_ones = NewTemp();
  Emit(all_ones + " = arith.constant -1 : " + type_str);
  std::string result = NewTemp();
  Emit(result + " = arith.xori " + src + ", " + all_ones + " : " + type_str);
  current_expr_value_ = result;
}

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

std::string PTOCodegen::GetScalarIterArgTypeString(
    const std::shared_ptr<const ScalarType>& scalar_type) const {
  if (scalar_type && scalar_type->dtype_ == DataType::BOOL) {
    return "i1";
  }
  if (scalar_type && scalar_type->dtype_.IsFloat()) {
    return GetTypeString(scalar_type->dtype_);
  }
  return "index";
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
      std::string ret_name = NewNamedTemp(return_var->name_hint_);
      BindVarToMlir(return_var, ret_name);
      return_var_names.push_back(ret_name);
      if (auto tensor_type = As<TensorType>(return_var->GetType())) {
        BindTensorView(return_var, ret_name);
        return_var_types.push_back(GetTensorViewTypeString(tensor_type.get()));
      } else if (auto tile_type = As<TileType>(return_var->GetType())) {
        INTERNAL_CHECK(tile_type->memref_.has_value())
            << "TileType return_var must have a MemRef at codegen stage for var: " << return_var->name_hint_;
        return_var_types.push_back(GetTileBufTypeString(tile_type->memref_.value().get()));
      } else {
        return_var_types.push_back(GetScalarIterArgTypeString(As<ScalarType>(return_var->GetType())));
      }
    }

    CHECK(op->else_body_.has_value()) << "IfStmt with return_vars requires else_body";

    // Emit: %ret0, %ret1 = scf.if %cond -> (type0, type1) {
    Emit(JoinCommaSep(return_var_names) + " = scf.if " + condition + " -> (" +
         JoinCommaSep(return_var_types) + ") {");
    indent_level_++;

    // Then branch
    yield_buffer_.clear();
    VisitStmt(op->then_body_);
    if (!yield_buffer_.empty()) {
      Emit("scf.yield " + JoinCommaSep(yield_buffer_) + " : " + JoinCommaSep(return_var_types));
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
        Emit("scf.yield " + JoinCommaSep(yield_buffer_) + " : " + JoinCommaSep(return_var_types));
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

  if (op->kind_ == ir::ForKind::Unroll) {
    LOG_WARN << "ForKind::Unroll loop was not expanded before codegen; "
                "generating sequential loop as fallback";
  }

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
  std::string loop_var_name = NewNamedTemp(op->loop_var_->name_hint_);
  BindVarToMlir(op->loop_var_, loop_var_name);

  // In PTO, only scalar types (index, f32, bool, etc.) need iter_args/yield
  // for loop-carried value semantics. Non-scalar types (TileType, TensorType)
  // are mutable references written in-place via outs(), so they are mapped
  // directly to their init values and excluded from iter_args/yield.
  std::vector<bool> is_scalar(op->iter_args_.size(), false);
  bool has_scalar_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (As<ScalarType>(op->iter_args_[i]->GetType())) {
      is_scalar[i] = true;
      has_scalar_iter_args = true;
    }
  }

  // Map non-scalar iter_args/return_vars directly to their init values
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (is_scalar[i]) continue;

    const auto& iter_arg = op->iter_args_[i];
    const auto& return_var = op->return_vars_[i];

    std::string init_mlir_name;
    auto tensor_type = As<TensorType>(iter_arg->GetType());
    if (tensor_type) {
      auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
      INTERNAL_CHECK(init_var) << "TensorType iter_arg init value must be a Var or IterArg";
      init_mlir_name = GetOrCreateTensorView(init_var);
    } else {
      VisitExpr(iter_arg->initValue_);
      init_mlir_name = current_expr_value_;
      current_expr_value_ = "";
    }

    BindVarToMlir(iter_arg, init_mlir_name);
    BindVarToMlir(return_var, init_mlir_name);

    if (tensor_type) {
      BindTensorView(iter_arg, init_mlir_name);
      BindTensorView(return_var, init_mlir_name);
    } else if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      const auto memref = ir::GetDefinedMemRef(tile_type);
      BindVarToMemRef(iter_arg, memref.get());
      BindVarToMemRef(return_var, memref.get());
    }
  }

  if (!has_scalar_iter_args) {
    // Simple scf.for (no iter_args, or all iter_args are non-scalar)
    Emit("scf.for " + loop_var_name + " = " + start + " to " + stop + " step " + step + " {");
    indent_level_++;

    yield_buffer_.clear();
    VisitStmt(op->body_);
    yield_buffer_.clear();

    indent_level_--;
    Emit("}");
  } else {
    // scf.for with scalar iter_args only
    std::vector<std::string> init_values;
    std::vector<std::string> iter_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!is_scalar[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(current_expr_value_);
      current_expr_value_ = "";

      std::string iter_name = NewNamedTemp(iter_arg->name_hint_);
      BindVarToMlir(iter_arg, iter_name);
      iter_arg_names.push_back(iter_name);

      iter_arg_types.push_back(GetScalarIterArgTypeString(As<ScalarType>(iter_arg->GetType())));
    }

    // Register return_vars SSA names (scalar only)
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!is_scalar[i]) continue;
      std::string ret_name = NewNamedTemp(op->return_vars_[i]->name_hint_);
      BindVarToMlir(op->return_vars_[i], ret_name);
      return_var_names.push_back(ret_name);
    }

    // Emit: %ret0 = scf.for %i = %start to %stop step %step
    //           iter_args(%acc = %init) -> (type) {
    Emit(JoinCommaSep(return_var_names) + " = scf.for " + loop_var_name + " = " + start + " to " + stop +
         " step " + step + " iter_args(" + JoinPairs(iter_arg_names, " = ", init_values) + ") -> (" +
         JoinCommaSep(iter_arg_types) + ") {");
    indent_level_++;

    yield_buffer_.clear();
    VisitStmt(op->body_);

    // Filter yield_buffer to keep only scalar iter_arg entries
    std::vector<std::string> scalar_yields;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (is_scalar[i] && i < yield_buffer_.size()) {
        scalar_yields.push_back(yield_buffer_[i]);
      }
    }

    // Emit scf.yield from filtered yield values
    if (!scalar_yields.empty()) {
      std::ostringstream yield_oss;
      yield_oss << "scf.yield ";
      for (size_t i = 0; i < scalar_yields.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << scalar_yields[i];
      }
      yield_oss << " : ";
      for (size_t i = 0; i < iter_arg_types.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << iter_arg_types[i];
      }
      Emit(yield_oss.str());
    }
    CHECK(scalar_yields.size() == iter_arg_types.size())
        << "ForStmt scalar yield count (" << scalar_yields.size() << ") must match scalar iter_args ("
        << iter_arg_types.size() << ")";
    yield_buffer_.clear();

    indent_level_--;
    Emit("}");
  }
}

void PTOCodegen::VisitStmt_(const WhileStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null WhileStmt";
  INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: WhileStmt has null condition";
  INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: WhileStmt has null body";

  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "WhileStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  // In PTO, only scalar types (index, f32, bool, etc.) need iter_args/yield
  // for loop-carried value semantics. Non-scalar types (TileType, TensorType)
  // are mutable references written in-place via outs(), so they are mapped
  // directly to their init values and excluded from iter_args/yield.
  std::vector<bool> is_scalar(op->iter_args_.size(), false);
  bool has_scalar_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (As<ScalarType>(op->iter_args_[i]->GetType())) {
      is_scalar[i] = true;
      has_scalar_iter_args = true;
    }
  }

  // Map non-scalar iter_args/return_vars directly to their init values
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (is_scalar[i]) continue;

    const auto& iter_arg = op->iter_args_[i];
    const auto& return_var = op->return_vars_[i];

    std::string init_mlir_name;
    auto tensor_type = As<TensorType>(iter_arg->GetType());
    if (tensor_type) {
      auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
      INTERNAL_CHECK(init_var) << "TensorType iter_arg init value must be a Var or IterArg";
      init_mlir_name = GetOrCreateTensorView(init_var);
    } else {
      VisitExpr(iter_arg->initValue_);
      init_mlir_name = current_expr_value_;
      current_expr_value_ = "";
    }

    BindVarToMlir(iter_arg, init_mlir_name);
    BindVarToMlir(return_var, init_mlir_name);

    if (tensor_type) {
      BindTensorView(iter_arg, init_mlir_name);
      BindTensorView(return_var, init_mlir_name);
    } else if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      const auto memref = ir::GetDefinedMemRef(tile_type);
      BindVarToMemRef(iter_arg, memref.get());
      BindVarToMemRef(return_var, memref.get());
    }
  }

  if (!has_scalar_iter_args) {
    // Simple scf.while (no iter_args, or all iter_args are non-scalar)
    Emit("scf.while : () -> () {");
    indent_level_++;

    VisitExpr(op->condition_);
    std::string cond = current_expr_value_;
    current_expr_value_ = "";
    Emit("scf.condition(" + cond + ")");

    indent_level_--;
    Emit("} do {");
    indent_level_++;

    yield_buffer_.clear();
    VisitStmt(op->body_);

    Emit("scf.yield");
    yield_buffer_.clear();

    indent_level_--;
    Emit("}");
  } else {
    // scf.while with scalar iter_args only
    std::vector<std::string> init_values;
    std::vector<std::string> before_arg_names;
    std::vector<std::string> after_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!is_scalar[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(current_expr_value_);
      current_expr_value_ = "";

      before_arg_names.push_back(NewTemp());
      after_arg_names.push_back(NewTemp());

      iter_arg_types.push_back(GetScalarIterArgTypeString(As<ScalarType>(iter_arg->GetType())));
    }

    // Register return_vars SSA names (scalar only)
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!is_scalar[i]) continue;

      std::string ret_name = NewTemp();
      BindVarToMlir(op->return_vars_[i], ret_name);
      return_var_names.push_back(ret_name);
    }

    // Lambda to register scalar iter_args in var_to_mlir_
    auto register_scalar_iter_args = [&](const std::vector<std::string>& ssa_names) {
      size_t scalar_idx = 0;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        if (!is_scalar[i]) continue;
        BindVarToMlir(op->iter_args_[i], ssa_names[scalar_idx]);
        scalar_idx++;
      }
    };

    std::string types_str = "(" + JoinCommaSep(iter_arg_types) + ")";

    // Emit: %ret0, %ret1 = scf.while (%before0 = %init0, ...) : (types) -> (types) {
    Emit(JoinCommaSep(return_var_names) + " = scf.while (" + JoinPairs(before_arg_names, " = ", init_values) +
         ") : " + types_str + " -> " + types_str + " {");
    indent_level_++;

    // Before region: register before-region args, evaluate condition
    register_scalar_iter_args(before_arg_names);

    VisitExpr(op->condition_);
    std::string cond = current_expr_value_;
    current_expr_value_ = "";

    // Emit: scf.condition(%cond) %before0, %before1 : type0, type1
    Emit("scf.condition(" + cond + ") " + JoinCommaSep(before_arg_names) + " : " +
         JoinCommaSep(iter_arg_types));

    indent_level_--;
    Emit("} do {");

    // After region: emit ^bb0 block header with typed arguments
    Emit("^bb0(" + JoinPairs(after_arg_names, " : ", iter_arg_types) + "):");
    indent_level_++;

    // Re-register iter_args with after-region SSA names
    register_scalar_iter_args(after_arg_names);

    // Visit body
    yield_buffer_.clear();
    VisitStmt(op->body_);

    // Filter yield_buffer to keep only scalar iter_arg entries
    std::vector<std::string> scalar_yields;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (is_scalar[i] && i < yield_buffer_.size()) {
        scalar_yields.push_back(yield_buffer_[i]);
      }
    }

    // Emit scf.yield from filtered yield values
    if (!scalar_yields.empty()) {
      Emit("scf.yield " + JoinCommaSep(scalar_yields) + " : " + JoinCommaSep(iter_arg_types));
    }
    CHECK(scalar_yields.size() == iter_arg_types.size())
        << "WhileStmt scalar yield count (" << scalar_yields.size() << ") must match scalar iter_args ("
        << iter_arg_types.size() << ")";
    yield_buffer_.clear();

    indent_level_--;
    Emit("}");
  }
}

}  // namespace codegen
}  // namespace pypto
