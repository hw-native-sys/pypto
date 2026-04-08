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
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/codegen/pto/tpop_chain_reorder.h"
#include "pypto/core/dtype.h"
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

static std::pair<VarPtr, VarPtr> GetTileValidShapeVars(const std::shared_ptr<const ir::TileType>& tile_type) {
  VarPtr valid_row_var;
  VarPtr valid_col_var;
  if (!tile_type || !tile_type->tile_view_.has_value()) {
    return {valid_row_var, valid_col_var};
  }

  const auto& tile_view = tile_type->tile_view_.value();
  if (tile_view.valid_shape.size() >= 1) {
    valid_row_var = As<ir::Var>(tile_view.valid_shape[0]);
  }
  if (tile_view.valid_shape.size() >= 2) {
    valid_col_var = As<ir::Var>(tile_view.valid_shape[1]);
  }
  return {valid_row_var, valid_col_var};
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
  CHECK(type == backend::BackendType::Ascend910B || type == backend::BackendType::Ascend950)
      << "PTOCodegen requires Ascend910B or Ascend950 backend, but unknown type is configured";
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
  fs_.constants_section.str("");
  fs_.constants_section.clear();
  fs_.body_section.str("");
  fs_.body_section.clear();

  auto type = backend::GetBackendType();
  std::string target_arch;
  switch (type) {
    case backend::BackendType::Ascend950:
      target_arch = "a5";
      break;
    case backend::BackendType::Ascend910B:
      target_arch = "a2a3";
      break;
    default:
      CHECK(false) << "Unsupported backend type for PTO target_arch: " << static_cast<int>(type);
  }
  stream_ << "module attributes {pto.target_arch = \"" << target_arch << "\"} {\n";

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
  fs_.Reset();
  fs_.current_function = func;

  // Reserve %argN names upfront so NewNamedTemp never collides with them
  for (size_t i = 0; i < func->params_.size(); i++) {
    fs_.used_ssa_names.insert("arg" + std::to_string(i));
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
      fs_.used_ssa_names.insert("arg" + std::to_string(func->params_.size() + i));
    }
  }

  BuildVarToMemRefMapping(func);

  MemRefCollectorVisitor collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }

  // Still collect fs_.memref_to_tile_type for GetTileBufTypeString fallback paths
  fs_.memref_to_tile_type = collector.GetMemRefTileTypes();

  // Per-var SSA binding: each tile variable gets its own SSA name
  for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
    std::string ssa_name = NewNamedTemp(tile_var->name_hint_);
    BindVarToMlir(tile_var, ssa_name);

    // Pre-populate type so body visitors (e.g., tile.reshape no-op check)
    // can query it before per-variable alloc_tile emission runs.
    std::string type_str = GetTileBufTypeStringFromTileType(tile_type, HasFillpadConsumer(tile_var.get()));
    fs_.ssa_to_tile_buf_type[ssa_name] = type_str;

    auto memref = ir::GetDefinedMemRef(tile_type);

    // Also maintain fs_.memref_to_mlir for compatibility (first var per MemRef)
    if (fs_.memref_to_mlir.find(memref.get()) == fs_.memref_to_mlir.end()) {
      fs_.memref_to_mlir[memref.get()] = ssa_name;
    }
  }

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

  // Separate params into tensors and scalars for tensors-first dispatch order.
  // PTOParam dispatches args as [tensors..., scalars...] regardless of function
  // signature order, so the MLIR function signature must match that layout.
  std::vector<size_t> tensor_param_indices;
  std::vector<size_t> scalar_param_indices;
  for (size_t i = 0; i < func->params_.size(); i++) {
    if (As<TensorType>(func->params_[i]->GetType())) {
      tensor_param_indices.push_back(i);
    } else {
      scalar_param_indices.push_back(i);
    }
  }

  // Assign %argN names: tensors get indices 0..N_tensors-1, scalars get N_tensors..
  size_t scalar_start_idx = tensor_param_indices.size();
  std::set<const ir::Var*> param_keys;
  for (size_t j = 0; j < tensor_param_indices.size(); j++) {
    const auto& param = func->params_[tensor_param_indices[j]];
    BindVarToMlir(param, "%arg" + std::to_string(j));
    param_keys.insert(GetVarKey(param));
  }
  for (size_t j = 0; j < scalar_param_indices.size(); j++) {
    const auto& param = func->params_[scalar_param_indices[j]];
    BindVarToMlir(param, "%arg" + std::to_string(scalar_start_idx + j));
    param_keys.insert(GetVarKey(param));
  }

  // Emit signature: tensors first, then scalars
  bool first_param = true;
  for (size_t j = 0; j < tensor_param_indices.size(); j++) {
    if (!first_param) stream_ << ", ";
    first_param = false;
    const auto& param = func->params_[tensor_param_indices[j]];
    auto tensor_type = As<TensorType>(param->GetType());
    stream_ << "%arg" << j << ": !pto.ptr<" << GetTypeString(tensor_type->dtype_) << ">";
  }
  for (size_t j = 0; j < scalar_param_indices.size(); j++) {
    if (!first_param) stream_ << ", ";
    first_param = false;
    const auto& param = func->params_[scalar_param_indices[j]];
    stream_ << "%arg" << (scalar_start_idx + j) << ": ";
    if (auto scalar_type = As<ScalarType>(param->GetType())) {
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

  stream_ << ")";
  switch (func->func_type_) {
    case ir::FunctionType::AIC:
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<cube>}";
      break;
    case ir::FunctionType::AIV:
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<vector>}";
      break;
    default:
      // Other function types like InCore are not expected here and have no kernel_kind.
      break;
  }
  stream_ << " {\n";
  indent_level_++;
  fs_.constants_indent = GetIndent();

  // Pre-emit i64 address constants now that indent_level_ is set
  for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
    if (fs_.tpop_result_vars.count(tile_var.get()) > 0) continue;
    auto memref = ir::GetDefinedMemRef(tile_type);
    if (memref && As<ir::ConstInt>(memref->addr_)) {
      GetOrEmitI64Constant(As<ir::ConstInt>(memref->addr_)->value_);
    }
  }

  // Parameters are already bound; non-param tile vars are bound above in per-var SSA binding

  for (const auto& var : func->params_) {
    if (auto tensor_type = As<TensorType>(var->GetType())) {
      // Skip tensor view for GM slot buffer workspace parameter (raw pointer, no view needed)
      if (var->name_hint_ == "__gm_pipe_buffer") {
        RecordGMSlotBufferSSA(GetVarName(var));
        continue;
      }
      std::string tensor_view = NewNamedTemp(var->name_hint_ + "_view");
      BindTensorView(var, tensor_view);

      for (const auto& j : tensor_type->shape_) {
        if (As<ir::ConstInt>(j)) {
          GetOrEmitIndexConstant(GetConstIntValue(j));
        }
      }
      // Pre-emit stride constants: use explicit tensor_view_.stride if available,
      // otherwise fall back to shape-based stride computation.
      bool has_explicit_stride =
          tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty();
      if (has_explicit_stride) {
        for (const auto& s : tensor_type->tensor_view_->stride) {
          if (As<ir::ConstInt>(s)) {
            GetOrEmitIndexConstant(GetConstIntValue(s));
          }
        }
      } else if (tensor_type->shape_.size() == 2) {
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
  stream_ = std::move(fs_.body_section);

  if (func->body_) {
    if (!fs_.tpop_result_vars.empty()) {
      auto seq = As<ir::SeqStmts>(func->body_);
      if (seq) {
        auto reordered = ReorderTpopChains(seq->stmts_);
        for (const auto& stmt : reordered) {
          VisitStmt(stmt);
        }
      } else {
        VisitStmt(func->body_);
      }
    } else {
      VisitStmt(func->body_);
    }
  }

  std::string body_content = stream_.str();
  stream_ = std::move(saved_stream);

  stream_ << fs_.constants_section.str();
  EmitMakeTensorViews(func);
  EmitExtraAllocTiles();
  stream_ << body_content;
  stream_ << GetIndent() << "return\n";

  indent_level_--;
  stream_ << "  }\n";
}

std::vector<ir::StmtPtr> PTOCodegen::ReorderTpopChains(const std::vector<ir::StmtPtr>& stmts) const {
  return codegen::ReorderTpopChains(stmts, fs_.tpop_result_vars);
}

void PTOCodegen::BuildVarToMemRefMapping(const FunctionPtr& func) {
  class VarMemRefMapper : public ir::IRVisitor {
   public:
    std::map<const ir::Var*, const ir::MemRef*>& var_to_memref;
    std::map<const ir::MemRef*, std::string>& memref_to_var_name;
    std::vector<std::pair<VarPtr, std::shared_ptr<const TileType>>>& tile_var_allocs;
    std::map<const ir::Var*, TpopResultInfo>& tpop_result_vars;
    std::set<const ir::Var*>& fillpad_input_vars;

    VarMemRefMapper(std::map<const ir::Var*, const ir::MemRef*>& mapping,
                    std::map<const ir::MemRef*, std::string>& reverse_mapping,
                    std::vector<std::pair<VarPtr, std::shared_ptr<const TileType>>>& allocs,
                    std::map<const ir::Var*, TpopResultInfo>& tpop_vars,
                    std::set<const ir::Var*>& fillpad_vars)
        : var_to_memref(mapping),
          memref_to_var_name(reverse_mapping),
          tile_var_allocs(allocs),
          tpop_result_vars(tpop_vars),
          fillpad_input_vars(fillpad_vars) {}

    void VisitStmt_(const AssignStmtPtr& op) override {
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        const auto memref = ir::GetDefinedMemRef(tile_type);
        const ir::MemRef* ptr = memref.get();
        var_to_memref[op->var_.get()] = ptr;
        if (memref_to_var_name.find(ptr) == memref_to_var_name.end()) {
          memref_to_var_name[ptr] = op->var_->name_hint_;
        }
        tile_var_allocs.emplace_back(op->var_, tile_type);

        if (auto call = As<ir::Call>(op->value_)) {
          // Track tpop result vars with their split value so codegen can:
          // 1. Skip alloc_tile for them
          // 2. Propagate split to tfree
          if (call->op_->name_ == "tile.tpop_from_aiv" || call->op_->name_ == "tile.tpop_from_aic") {
            int split = call->GetKwarg<int>("split", 0);
            tpop_result_vars[op->var_.get()] = TpopResultInfo{split, call->op_->name_};
          }
          // Track fillpad input variables so we know which tiles need
          // physical dims on alloc_tile + set_validshape after tload.
          if (call->op_->name_ == "tile.fillpad" && !call->args_.empty()) {
            if (auto input_var = As<ir::Var>(call->args_[0])) {
              fillpad_input_vars.insert(input_var.get());
            }
          }
        }
      }
      ir::IRVisitor::VisitStmt_(op);
    }
  };

  VarMemRefMapper mapper(fs_.var_to_memref, fs_.memref_to_var_name, fs_.tile_var_allocs, fs_.tpop_result_vars,
                         fs_.fillpad_input_vars);
  if (func->body_) {
    mapper.VisitStmt(func->body_);
  }
}

void PTOCodegen::EmitMakeTensorViews(const FunctionPtr& func) {
  for (const auto& param : func->params_) {
    if (auto tensor_type = As<TensorType>(param->GetType())) {
      // Skip GM slot buffer workspace parameter (raw pointer, no view needed)
      if (param->name_hint_ == "__gm_pipe_buffer") continue;

      std::string tensor_view = fs_.tensor_to_view.at(GetVarKey(param));

      bool layout_DN = false;
      if (tensor_type->tensor_view_.has_value()) {
        if (tensor_type->tensor_view_.value().layout == ir::TensorLayout::DN) {
          layout_DN = true;
        }
      }

      // [M, 1] column vectors: PTOAS always infers DN for shape [M, 1] with
      // degenerate strides, so force DN layout and emit strides [1, M].
      bool is_column_vector = false;
      if (tensor_type->shape_.size() == 2) {
        auto last_dim = As<ir::ConstInt>(tensor_type->shape_[1]);
        if (last_dim && last_dim->value_ == 1) {
          is_column_vector = true;
          layout_DN = true;
        }
      }

      // Check if tensor_view_ provides explicit strides (e.g. for view output tensors
      // whose physical memory layout differs from the view shape).
      bool has_explicit_stride =
          tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty();

      // For N-D (N > 2): pre-compute row-major strides as SSA values using arith.muli
      // so that dynamic dimensions (ir::Var) are handled correctly. Emit any needed
      // multiply instructions BEFORE the make_tensor_view line.
      // Skip when explicit strides are available.
      std::vector<std::string> nd_stride_names;
      if (!has_explicit_stride && tensor_type->shape_.size() > 2) {
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
      stream_ << GetVarName(param);

      stream_ << ", shape = [";
      if (layout_DN && tensor_type->shape_.size() == 2 && !is_column_vector) {
        // General DN 2D: emit transposed shape [shape[1], shape[0]] so that PTOAS
        // sees the column-major convention while the IR keeps original shape.
        for (int j = 1; j >= 0; j--) {
          if (j < 1) stream_ << ", ";
          if (auto var = As<ir::Var>(tensor_type->shape_[j])) {
            stream_ << GetVarName(var);
          } else {
            stream_ << GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[j]));
          }
        }
      } else {
        for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
          if (j > 0) stream_ << ", ";
          if (auto var = As<ir::Var>(tensor_type->shape_[j])) {
            stream_ << GetVarName(var);
          } else {
            stream_ << GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[j]));
          }
        }
      }
      stream_ << "],";

      stream_ << " strides = [";
      if (has_explicit_stride) {
        // Use explicit strides from tensor_view_ (e.g. physical memory strides for view tensors)
        const auto& strides = tensor_type->tensor_view_->stride;
        for (size_t j = 0; j < strides.size(); j++) {
          if (j > 0) stream_ << ", ";
          if (auto var = As<ir::Var>(strides[j])) {
            stream_ << GetVarName(var);
          } else {
            stream_ << GetOrEmitIndexConstant(GetConstIntValue(strides[j]));
          }
        }
      } else if (tensor_type->shape_.size() == 2) {
        // For column vector [M, 1]: stride dim is shape[0] (= M) → strides [1, M].
        // For other 2D: stride dim is shape[1] (= C) → DN [1, C] or ND [C, 1].
        int stride_idx = is_column_vector ? 0 : 1;
        std::string row_stride;
        if (auto var = As<ir::Var>(tensor_type->shape_[stride_idx])) {
          row_stride = GetVarName(var);
        } else {
          row_stride = GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[stride_idx]));
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

      std::string layout_str = "nd";
      if (is_column_vector) {
        layout_str = "dn";
      } else if (tensor_type->tensor_view_.has_value()) {
        switch (tensor_type->tensor_view_.value().layout) {
          case ir::TensorLayout::DN:
            layout_str = "dn";
            break;
          case ir::TensorLayout::NZ:
            layout_str = "nz";
            break;
          case ir::TensorLayout::ND:
            break;
        }
      }
      stream_ << " {layout = #pto.layout<" << layout_str << ">}";

      stream_ << ": !pto.tensor_view<";
      for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
        if (j > 0) stream_ << "x";
        stream_ << "?";
      }
      stream_ << "x" << GetTypeString(tensor_type->dtype_) << ">\n";
    }
  }
}

void PTOCodegen::EmitAllocTileForVar(const ir::VarPtr& tile_var,
                                     const std::shared_ptr<const ir::TileType>& tile_type) {
  auto var_key = GetVarKey(tile_var);
  if (!fs_.emitted_tile_alloc_vars.insert(var_key).second) {
    return;
  }

  auto mlir_it = fs_.var_to_mlir.find(var_key);
  INTERNAL_CHECK(mlir_it != fs_.var_to_mlir.end())
      << "Tile var " << tile_var->name_hint_ << " not found in fs_.var_to_mlir";
  std::string tile_buf = mlir_it->second;

  // Generate type string first — ExtractTileTypeInfo already decides v_row=?/v_col=?.
  // For tiles consumed by fillpad, force ALL dynamic dims (pto.set_validshape requires both ?).
  bool has_fillpad = HasFillpadConsumer(tile_var.get());
  std::string type_str = GetTileBufTypeStringFromTileType(tile_type, has_fillpad);
  bool type_is_dynamic =
      (type_str.find("v_row=?") != std::string::npos || type_str.find("v_col=?") != std::string::npos);

  std::string valid_row_mlir;
  std::string valid_col_mlir;
  if (tile_type->tile_view_.has_value()) {
    const auto& tv = tile_type->tile_view_.value();
    bool has_pad = (tv.pad != ir::PadValue::null);
    if (!has_pad && type_is_dynamic) {
      // Check if this tile is consumed by fillpad.
      // If yes: use physical dims so TLOAD DMA uses correct stride; set_validshape sets actual region.
      // If no: use dynamic variable as operand (TLOAD respects valid_shape for DMA).
      if (has_fillpad) {
        if (tile_type->shape_.size() >= 1) {
          if (auto c = As<ir::ConstInt>(tile_type->shape_[0])) {
            valid_row_mlir = GetOrEmitIndexConstant(c->value_);
          }
        }
        if (tile_type->shape_.size() >= 2) {
          if (auto c = As<ir::ConstInt>(tile_type->shape_[1])) {
            valid_col_mlir = GetOrEmitIndexConstant(c->value_);
          }
        }
      } else {
        // No fillpad: use dynamic variable as operand (old behavior).
        auto [valid_row_var, valid_col_var] = GetTileValidShapeVars(tile_type);
        if (valid_row_var) valid_row_mlir = GetVarName(valid_row_var);
        if (valid_col_var) valid_col_mlir = GetVarName(valid_col_var);
      }
    }
    // Static v_row/v_col: type string already encodes the values (e.g. v_row=48).
    // PTOAS requires valid_row/valid_col operands to be ABSENT when static.
  }
  auto memref = ir::GetDefinedMemRef(tile_type);
  std::string addr_ssa;
  if (memref) {
    if (auto const_addr = As<ir::ConstInt>(memref->addr_)) {
      addr_ssa = GetOrEmitI64Constant(const_addr->value_);
    }
  }

  std::ostringstream line;
  line << tile_buf << " = pto.alloc_tile";
  if (!addr_ssa.empty()) line << " addr = " << addr_ssa;
  if (!valid_row_mlir.empty()) line << " valid_row = " << valid_row_mlir;
  if (!valid_col_mlir.empty()) line << " valid_col = " << valid_col_mlir;
  line << " : " << type_str;
  Emit(line.str());

  fs_.ssa_to_tile_buf_type[tile_buf] = type_str;
}

// ========================================================================
// Private helpers
// ========================================================================

std::string PTOCodegen::GetIndent() const { return std::string(static_cast<size_t>(indent_level_) * 2, ' '); }

std::string PTOCodegen::GetOrEmitIndexConstant(int64_t value) {
  auto it = fs_.emitted_constants.find(value);
  if (it != fs_.emitted_constants.end()) {
    return it->second;
  }
  std::string ssa_id = "c" + std::to_string(value);
  std::string name;
  if (fs_.used_ssa_names.find(ssa_id) == fs_.used_ssa_names.end()) {
    fs_.used_ssa_names.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }
  fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << value << " : index\n";
  fs_.emitted_constants[value] = name;
  return name;
}

std::string PTOCodegen::GetOrEmitI64Constant(int64_t value) {
  auto it = fs_.emitted_i64_constants.find(value);
  if (it != fs_.emitted_i64_constants.end()) {
    return it->second;
  }
  std::string ssa_id;
  if (value == 0) {
    ssa_id = "c0i";
  } else if (value < 0) {
    uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
    ssa_id = "cn" + std::to_string(magnitude);
  } else {
    ssa_id = "c" + std::to_string(value);
  }
  std::string name;
  if (fs_.used_ssa_names.find(ssa_id) == fs_.used_ssa_names.end()) {
    fs_.used_ssa_names.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }
  fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << value << " : i64\n";
  fs_.emitted_i64_constants[value] = name;
  return name;
}

std::string PTOCodegen::GetOrEmitI32Constant(int32_t value) {
  auto it = fs_.emitted_i32_constants.find(value);
  if (it != fs_.emitted_i32_constants.end()) {
    return it->second;
  }
  std::string ssa_id;
  if (value == 0) {
    ssa_id = "c0_i32";
  } else if (value < 0) {
    uint32_t magnitude = static_cast<uint32_t>(-(value + 1)) + 1;
    ssa_id = "cn" + std::to_string(magnitude) + "_i32";
  } else {
    ssa_id = "c" + std::to_string(value) + "_i32";
  }
  std::string name;
  if (fs_.used_ssa_names.find(ssa_id) == fs_.used_ssa_names.end()) {
    fs_.used_ssa_names.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }
  fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << value << " : i32\n";
  fs_.emitted_i32_constants[value] = name;
  return name;
}

std::string PTOCodegen::GetTileBufForMemRef(const MemRefPtr& memref) const {
  auto it = fs_.memref_to_mlir.find(memref.get());
  INTERNAL_CHECK(it != fs_.memref_to_mlir.end()) << "MemRef not found in mapping";
  return it->second;
}

std::string PTOCodegen::AllocNewTileBuf(const std::string& tile_buf_type_string, const std::string& name_hint,
                                        const std::string& addr_ssa, const std::string& valid_row_ssa,
                                        const std::string& valid_col_ssa) {
  std::string name = NewNamedTemp(name_hint);
  fs_.extra_alloc_tiles.push_back(
      FunctionState::ExtraAllocTile{name, tile_buf_type_string, addr_ssa, valid_row_ssa, valid_col_ssa});
  fs_.ssa_to_tile_buf_type[name] = tile_buf_type_string;
  return name;
}

void PTOCodegen::SetCurrentResultBuf(const std::string& buf) { fs_.current_result_buf = buf; }

void PTOCodegen::RegisterTileBufType(const std::string& ssa_name, const std::string& type_string) {
  fs_.ssa_to_tile_buf_type[ssa_name] = type_string;
}

std::string PTOCodegen::GetSSATileBufType(const std::string& ssa_name) const {
  auto it = fs_.ssa_to_tile_buf_type.find(ssa_name);
  return it != fs_.ssa_to_tile_buf_type.end() ? it->second : std::string{};
}

void PTOCodegen::RecordGMSlotBufferSSA(const std::string& ssa) { fs_.gm_slot_buffer_ssa = ssa; }

std::string PTOCodegen::GetGMSlotBufferSSA() const { return fs_.gm_slot_buffer_ssa; }

int PTOCodegen::GetValidatedTpopSplit(const ir::Var* var, const std::string& expected_tpop_op_name,
                                      const std::string& tfree_op_name) const {
  auto it = fs_.tpop_result_vars.find(var);
  INTERNAL_CHECK(it != fs_.tpop_result_vars.end())
      << "Internal error: GetValidatedTpopSplit called for var not in fs_.tpop_result_vars";
  CHECK(it->second.op_name == expected_tpop_op_name)
      << tfree_op_name << " requires its tile argument to come from " << expected_tpop_op_name << ", got "
      << it->second.op_name;
  return it->second.split;
}

bool PTOCodegen::IsAICFunction() const {
  return fs_.current_function && fs_.current_function->func_type_ == ir::FunctionType::AIC;
}

bool PTOCodegen::IsAIVFunction() const {
  return fs_.current_function && fs_.current_function->func_type_ == ir::FunctionType::AIV;
}

void PTOCodegen::EmitExtraAllocTiles() {
  for (const auto& alloc : fs_.extra_alloc_tiles) {
    stream_ << GetIndent() << alloc.name << " = pto.alloc_tile";
    if (!alloc.addr_ssa.empty()) {
      stream_ << " addr = " << alloc.addr_ssa;
    }
    if (!alloc.valid_row_ssa.empty()) {
      stream_ << " valid_row = " << alloc.valid_row_ssa;
    }
    if (!alloc.valid_col_ssa.empty()) {
      stream_ << " valid_col = " << alloc.valid_col_ssa;
    }
    stream_ << " : " << alloc.type_string << "\n";
  }
}

// ========================================================================
// Statement visitors
// ========================================================================

void PTOCodegen::VisitStmt_(const AssignStmtPtr& op) {
  if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
    if (fs_.tpop_result_vars.count(op->var_.get()) == 0) {
      EmitAllocTileForVar(op->var_, tile_type);
    }
  }

  if (auto call = As<ir::Call>(op->value_)) {
    if (backend_ != nullptr && backend_->GetOpInfo(call->op_->name_) != nullptr) {
      std::string result_buf =
          op->var_->name_hint_;  // Seed for readable MLIR names when no tile buffer exists.
      std::shared_ptr<const TileType> result_tile_type;
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        // Prefer per-var SSA name from fs_.var_to_mlir (set during per-var alloc binding)
        auto var_it = fs_.var_to_mlir.find(GetVarKey(op->var_));
        if (var_it != fs_.var_to_mlir.end()) {
          result_buf = var_it->second;
        } else {
          result_buf = GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
        }
        result_tile_type = tile_type;
      } else if (auto tile_type = As<TileType>(op->var_->GetType())) {
        result_tile_type = tile_type;
      } else {
        // Pre-allocate a %-prefixed SSA name for non-tile backend ops (e.g., scalar
        // results like tile.getval, or i32 results like reserve_buffer / import_peer_buffer).
        // Register it in fs_.var_to_mlir so subsequent expressions can resolve the variable.
        result_buf = NewNamedTemp(op->var_->name_hint_);
        BindVarToMlir(op->var_, result_buf);
      }
      fs_.current_result_var = op->var_;
      fs_.current_result_buf = result_buf;
      fs_.current_result_tile_type = result_tile_type;
      VisitExpr(op->value_);
      // If codegen changed the result buffer (e.g., reshape allocated a new tile),
      // update variable mapping so subsequent references use the new buffer
      if (!fs_.current_result_buf.empty() && fs_.current_result_buf != result_buf) {
        BindVarToMlir(op->var_, fs_.current_result_buf);
      }
      // Register per-variable tile_buf type from the variable's own TileType.
      // This ensures that even when multiple variables share a MemRef, each
      // variable's SSA value carries its correct typed annotation.
      if (result_tile_type && !fs_.current_result_buf.empty()) {
        bool fillpad_force = HasFillpadConsumer(op->var_.get());
        std::string var_type_str = GetTileBufTypeStringFromTileType(result_tile_type, fillpad_force);
        if (!var_type_str.empty()) {
          fs_.ssa_to_tile_buf_type[fs_.current_result_buf] = var_type_str;
        }
      }
      fs_.current_result_var.reset();
      fs_.current_result_buf.clear();
      fs_.current_result_tile_type = nullptr;
      return;
    }
  }

  fs_.current_expr_value = "";
  VisitExpr(op->value_);
  // Register scalar/index result so subsequent expressions can look up this variable
  if (As<ScalarType>(op->var_->GetType()) && !fs_.current_expr_value.empty()) {
    BindVarToMlir(op->var_, fs_.current_expr_value);
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

std::string PTOCodegen::GetCurrentResultTarget() const { return fs_.current_result_buf; }

ir::VarPtr PTOCodegen::GetCurrentResultVar() const { return fs_.current_result_var; }

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
  fs_.current_expr_value = "";
  VisitExpr(expr);
  std::string result = fs_.current_expr_value;
  fs_.current_expr_value = "";
  if (!result.empty()) {
    return result;
  }

  LOG_ERROR << "GetExprAsCode for unsupported expression type";
  return "";
}

std::string PTOCodegen::GetTypeString(const DataType& dtype) const { return DataTypeToMLIR(dtype); }

const ir::Var* PTOCodegen::GetVarKey(const VarPtr& var) const {
  INTERNAL_CHECK(var != nullptr) << "Internal error: variable key requested for null Var";
  return var.get();
}

void PTOCodegen::BindVarToMlir(const VarPtr& var, const std::string& mlir_name) {
  fs_.var_to_mlir[GetVarKey(var)] = mlir_name;
}

void PTOCodegen::BindTensorView(const VarPtr& var, const std::string& tensor_view_name) {
  fs_.tensor_to_view[GetVarKey(var)] = tensor_view_name;
}

void PTOCodegen::BindVarToMemRef(const VarPtr& var, const ir::MemRef* memref) {
  fs_.var_to_memref[GetVarKey(var)] = memref;
}

std::string PTOCodegen::GetVarName(const VarPtr& var) const {
  auto key = GetVarKey(var);
  auto it = fs_.var_to_mlir.find(key);
  if (it != fs_.var_to_mlir.end()) {
    return it->second;
  }
  auto memref_it = fs_.var_to_memref.find(key);
  if (memref_it != fs_.var_to_memref.end()) {
    auto mlir_it = fs_.memref_to_mlir.find(memref_it->second);
    if (mlir_it != fs_.memref_to_mlir.end()) {
      return mlir_it->second;
    }
  }
  if (auto tile_type = ir::GetTileTypeWithMemRef(var->GetType())) {
    return GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
  }
  for (const auto& [mapped_var, mlir_name] : fs_.var_to_mlir) {
    if (mapped_var && mapped_var->name_hint_ == var->name_hint_) {
      return mlir_name;
    }
  }
  LOG_ERROR << "Variable " << var->name_hint_ << " not found in MLIR mapping";
  return "";
}

std::string PTOCodegen::NewTemp() {
  std::string name = std::to_string(fs_.temp_counter++);
  while (fs_.used_ssa_names.count(name)) {
    name = std::to_string(fs_.temp_counter++);
  }
  fs_.used_ssa_names.insert(name);
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

  if (!sanitized.empty() && fs_.used_ssa_names.find(sanitized) == fs_.used_ssa_names.end()) {
    fs_.used_ssa_names.insert(sanitized);
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

int64_t PTOCodegen::GetConstIntValue(const ExprPtr& expr) const {
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return const_int->value_;
  }
  LOG_ERROR << "Expected ConstInt expression";
  return 0;
}

std::string PTOCodegen::GetOrCreateTensorView(const VarPtr& tensor_var) {
  auto it = fs_.tensor_to_view.find(GetVarKey(tensor_var));
  if (it != fs_.tensor_to_view.end()) return it->second;
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
  if (fs_.emitted_float_constants.find(value) == fs_.emitted_float_constants.end()) {
    std::string ssa_id = "cst";
    if (!fs_.emitted_float_constants.empty()) {
      ssa_id += "_" + std::to_string(fs_.emitted_float_constants.size());
    }
    std::string name;
    if (fs_.used_ssa_names.find(ssa_id) == fs_.used_ssa_names.end()) {
      fs_.used_ssa_names.insert(ssa_id);
      name = "%" + ssa_id;
    } else {
      name = NewTemp();
    }

    std::ostringstream val_str;
    val_str << std::scientific << std::setprecision(6) << value;

    fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << val_str.str() << " : "
                          << mlir_type << "\n";
    fs_.emitted_float_constants.insert(value);
    fs_.float_const_names[value] = name;
    return name;
  }
  return fs_.float_const_names[value];
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
  auto tile_it = fs_.memref_to_tile_type.find(memref);
  INTERNAL_CHECK(tile_it != fs_.memref_to_tile_type.end())
      << "Internal error: missing tile type for MemRef '" << memref->name_hint_ << "'";
  auto memory_space = tile_it->second->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value()) << "Internal error: tile type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  auto c = ExtractTileTypeInfo(*tile_it->second, GetTypeString(tile_it->second->dtype_));
  return FormatTileBufTypeString(loc, c.dtype_str, c.rows, c.cols, c.blayout, c.slayout, c.fractal, c.pad,
                                 c.v_row, c.v_col, c.v_row_dynamic, c.v_col_dynamic);
}

std::string PTOCodegen::GetTileBufTypeStringFromTileType(const std::shared_ptr<const ir::TileType>& tile_type,
                                                         bool force_all_dynamic) const {
  INTERNAL_CHECK(tile_type) << "Internal error: tile_type must not be null";
  auto memory_space = tile_type->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value()) << "Internal error: tile_type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  auto c = ExtractTileTypeInfo(*tile_type, GetTypeString(tile_type->dtype_), force_all_dynamic);
  return FormatTileBufTypeString(loc, c.dtype_str, c.rows, c.cols, c.blayout, c.slayout, c.fractal, c.pad,
                                 c.v_row, c.v_col, c.v_row_dynamic, c.v_col_dynamic);
}

std::string PTOCodegen::GetExprTypeAnnotation(const ir::ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    auto key = GetVarKey(var);
    // Primary lookup: SSA name → tile_buf type (covers root allocs AND view results)
    auto mlir_it = fs_.var_to_mlir.find(key);
    if (mlir_it != fs_.var_to_mlir.end()) {
      auto ssa_it = fs_.ssa_to_tile_buf_type.find(mlir_it->second);
      if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
        return ssa_it->second;
      }
    }
    // Per-variable TileType: derives the type from the variable's own
    // TileType, which is correct for view op results (slice, reshape,
    // fillpad) whose type differs from the root alloc's type.
    if (auto tile_type = As<TileType>(var->GetType())) {
      if (tile_type->memref_.has_value()) {
        return GetTileBufTypeStringFromTileType(tile_type);
      }
    }
    // Fallback: var → memref → root alloc type
    auto memref_it = fs_.var_to_memref.find(key);
    if (memref_it != fs_.var_to_memref.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    if (auto scalar_type = As<ScalarType>(var->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
  }
  if (auto iter_arg = As<ir::IterArg>(expr)) {
    auto key = GetVarKey(std::dynamic_pointer_cast<const ir::Var>(iter_arg));
    auto mlir_it = fs_.var_to_mlir.find(key);
    if (mlir_it != fs_.var_to_mlir.end()) {
      auto ssa_it = fs_.ssa_to_tile_buf_type.find(mlir_it->second);
      if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
        return ssa_it->second;
      }
    }
    if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      return GetTileBufTypeStringFromTileType(tile_type);
    }
    auto memref_it = fs_.var_to_memref.find(key);
    if (memref_it != fs_.var_to_memref.end()) {
      return GetTileBufTypeString(memref_it->second);
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
  // Prefer type registered by alloc_tile (may have force_all_dynamic for fillpad tiles)
  if (!fs_.current_result_buf.empty()) {
    auto ssa_it = fs_.ssa_to_tile_buf_type.find(fs_.current_result_buf);
    if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
      return ssa_it->second;
    }
  }
  if (fs_.current_result_tile_type && fs_.current_result_tile_type->memref_.has_value()) {
    return GetTileBufTypeString(fs_.current_result_tile_type->memref_.value().get());
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultTileBufTypeStringFromTileType() const {
  if (fs_.current_result_tile_type && fs_.current_result_tile_type->memref_.has_value()) {
    bool fillpad_force = fs_.current_result_var ? HasFillpadConsumer(fs_.current_result_var.get()) : false;
    return GetTileBufTypeStringFromTileType(fs_.current_result_tile_type, fillpad_force);
  }
  return "";
}

}  // namespace codegen
}  // namespace pypto
