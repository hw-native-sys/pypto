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

#include "pypto/codegen/pto/pto_ir_printer.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/pto_target_lowering.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::AssignStmt;
using ir::AssignStmtPtr;
using ir::AsTensorTypeLike;
using ir::AsVarLike;
using ir::AtomicType;
using ir::Call;
using ir::CallPtr;
using ir::ConstFloat;
using ir::ConstInt;
using ir::ConstIntPtr;
using ir::EvalStmt;
using ir::ExprPtr;
using ir::FunctionPtr;
using ir::FunctionType;
using ir::IsA;
using ir::IsInCoreType;
using ir::IsOp;
using ir::MakeTuple;
using ir::OpRegistry;
using ir::ProgramPtr;
using ir::PTOOperandRole;
using ir::PTOTileBufType;
using ir::ReturnStmt;
using ir::ScalarType;
using ir::Span;
using ir::StmtPtr;
using ir::structural_equal;
using ir::TensorLayout;
using ir::TensorType;
using ir::Var;
using ir::VarPtr;

namespace transform_utils = ir::transform_utils;

namespace {

using ConstantKey = std::pair<int64_t, int>;
using FloatConstantKey = std::pair<uint64_t, int>;

struct FloatConstantInfo {
  double value;
  DataType dtype;
  std::string name;
};

struct PrintedOperand {
  std::string value;
  std::string type;
};

struct ResolvedLogicalTensorView {
  TensorLayout layout;
  std::vector<ExprPtr> strides;
};

ResolvedLogicalTensorView ResolveLogicalTensorView(const TensorType& tensor) {
  const auto tensor_view = tensor.tensor_view_.value_or(ir::TensorView{});
  TensorLayout layout = tensor_view.layout;
  if (auto last = As<ConstInt>(tensor.shape_.back());
      tensor.shape_.size() >= 2 && last && last->value_ == 1) {
    layout = TensorLayout::DN;
  }

  std::vector<ExprPtr> strides = tensor_view.stride;
  if (strides.empty()) {
    strides = ir::tensor_view_semantics::BuildLogicalStridesFromLayout(tensor.shape_, layout);
  }
  return {layout, std::move(strides)};
}

uint64_t FloatBits(double value) {
  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::string SanitizeName(const std::string& name) {
  std::string result;
  result.reserve(name.size());
  for (char ch : name) {
    const auto uch = static_cast<unsigned char>(ch);
    result.push_back(std::isalnum(uch) || ch == '_' ? ch : '_');
  }
  if (result.empty()) result = "tmp";
  if (std::isdigit(static_cast<unsigned char>(result.front()))) result.insert(result.begin(), '_');
  return result;
}

std::string TensorViewTypeString(const TensorType& type) {
  std::ostringstream out;
  out << "!pto.tensor_view<";
  for (size_t i = 0; i < type.shape_.size(); ++i) {
    if (i > 0) out << "x";
    out << "?";
  }
  out << "x" << DataTypeToMLIR(type.dtype_) << ">";
  return out.str();
}

std::string PartitionTypeString(const MakeTuple& sizes, DataType dtype) {
  std::ostringstream out;
  out << "!pto.partition_tensor_view<";
  for (size_t i = 0; i < sizes.elements_.size(); ++i) {
    if (i > 0) out << "x";
    if (auto extent = As<ConstInt>(sizes.elements_[i])) {
      out << extent->value_;
    } else {
      out << "?";
    }
  }
  out << "x" << DataTypeToMLIR(dtype) << ">";
  return out.str();
}

std::string TileBufferTypeString(const PTOTileBufType& type) {
  return FormatTileBufTypeString(
      MemorySpaceToMLIR(type.memory_space_), DataTypeToMLIR(type.dtype_), type.rows_, type.cols_,
      type.blayout_, type.slayout_, type.fractal_, type.pad_, type.valid_rows_.value_or(type.rows_),
      type.valid_cols_.value_or(type.cols_), !type.valid_rows_.has_value(), !type.valid_cols_.has_value());
}

class FunctionPrinter final {
 public:
  FunctionPrinter(std::string target_arch, bool emit_tile_addr)
      : target_arch_(std::move(target_arch)), emit_tile_addr_(emit_tile_addr) {}

  std::string PrintProgram(const ProgramPtr& program) {
    INTERNAL_CHECK(program) << "PTOIRPrinter requires a non-null Program";
    stream_ << "module attributes {pto.target_arch = \"" << target_arch_ << "\"} {\n";
    for (const auto& [global_var, function] : program->functions_) {
      (void)global_var;
      PrintFunction(function);
    }
    stream_ << "}\n";
    return stream_.str();
  }

  std::string PrintSingleFunction(const FunctionPtr& function) {
    PrintFunction(function);
    return stream_.str();
  }

 private:
  void ResetFunctionState() {
    var_names_.clear();
    tensor_views_.clear();
    constants_.clear();
    float_constants_.clear();
    used_names_.clear();
    yield_types_.clear();
    next_temp_ = 0;
    indent_ = 2;
  }

  void PrintFunction(const FunctionPtr& function) {
    INTERNAL_CHECK(function) << "PTOIRPrinter requires non-null functions";
    INTERNAL_CHECK_SPAN(IsInCoreType(function->func_type_), function->span_)
        << "PTOIRPrinter supports only InCore-variant functions";
    ResetFunctionState();
    CollectConstants(function);
    BindParameters(function);

    stream_ << "  func.func @" << function->name_ << "(";
    PrintSignature(function);
    stream_ << ")";
    if (function->func_type_ == FunctionType::AIC) {
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<cube>}";
    } else if (function->func_type_ == FunctionType::AIV) {
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<vector>}";
    }
    stream_ << " {\n";

    EmitConstants();
    EmitTensorViews(function);
    for (const auto& stmt : transform_utils::FlattenToStmts(function->body_)) PrintStmt(stmt);
    stream_ << Indent() << "return\n";
    stream_ << "  }\n";
  }

  void BindParameters(const FunctionPtr& function) {
    std::vector<VarPtr> tensors;
    std::vector<VarPtr> scalars;
    for (const auto& param : function->params_) {
      if (AsTensorTypeLike(param->GetType())) {
        tensors.push_back(param);
      } else {
        scalars.push_back(param);
      }
    }
    size_t index = 0;
    for (const auto& param : tensors) {
      const std::string name = "arg" + std::to_string(index++);
      used_names_.insert(name);
      var_names_[param.get()] = "%" + name;
    }
    for (const auto& param : scalars) {
      const std::string name = ir::IsSyntheticPTOTargetParamName(param->name_hint_)
                                   ? param->name_hint_
                                   : "arg" + std::to_string(index++);
      used_names_.insert(name);
      var_names_[param.get()] = "%" + name;
    }
  }

  void PrintSignature(const FunctionPtr& function) {
    std::vector<VarPtr> ordered;
    for (const auto& param : function->params_) {
      if (AsTensorTypeLike(param->GetType())) ordered.push_back(param);
    }
    for (const auto& param : function->params_) {
      if (!AsTensorTypeLike(param->GetType())) ordered.push_back(param);
    }
    for (size_t i = 0; i < ordered.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << (ir::IsSyntheticPTOTargetParamName(ordered[i]->name_hint_) ? "%" + ordered[i]->name_hint_
                                                                            : "%arg" + std::to_string(i))
              << ": ";
      if (auto tensor = AsTensorTypeLike(ordered[i]->GetType())) {
        stream_ << "!pto.ptr<" << DataTypeToMLIR(tensor->dtype_) << ">";
      } else if (auto scalar = As<ScalarType>(ordered[i]->GetType())) {
        stream_ << DataTypeToMLIR(scalar->dtype_);
      } else {
        INTERNAL_CHECK_SPAN(false, ordered[i]->span_)
            << "PTOIRPrinter Step-4 slice supports only Tensor and Scalar parameters";
      }
    }
  }

  void CollectExprConstants(const ExprPtr& expr) {
    if (!expr) return;
    if (auto value = As<ConstInt>(expr)) {
      constants_.emplace(ConstantKey{value->value_, value->dtype().Code()}, value->dtype());
      return;
    }
    if (auto value = As<ConstFloat>(expr)) {
      const FloatConstantKey key{FloatBits(value->value_), value->dtype().Code()};
      if (float_constants_.find(key) == float_constants_.end()) {
        const std::string suffix =
            float_constants_.empty() ? "" : "_" + std::to_string(float_constants_.size());
        float_constants_.emplace(key, FloatConstantInfo{value->value_, value->dtype(), "%cst" + suffix});
      }
      return;
    }
    if (auto tuple = As<MakeTuple>(expr)) {
      for (const auto& element : tuple->elements_) CollectExprConstants(element);
      return;
    }
    if (auto binary = std::dynamic_pointer_cast<const ir::BinaryExpr>(expr)) {
      CollectExprConstants(binary->left_);
      CollectExprConstants(binary->right_);
      return;
    }
    if (auto cast = As<ir::Cast>(expr)) {
      CollectExprConstants(cast->operand_);
    }
  }

  void CollectConstants(const FunctionPtr& function) {
    for (const auto& param : function->params_) {
      auto tensor = AsTensorTypeLike(param->GetType());
      if (!tensor) continue;
      CollectTensorConstants(*tensor);
    }
    CollectStmtConstants(function->body_);
  }

  void CollectStmtConstants(const StmtPtr& body) {
    for (const auto& stmt : transform_utils::FlattenToStmts(body)) {
      CallPtr call;
      if (auto assign = As<AssignStmt>(stmt)) {
        if (auto tensor = AsTensorTypeLike(assign->var_->GetType())) CollectTensorConstants(*tensor);
        call = As<Call>(assign->value_);
        if (!call) CollectExprConstants(assign->value_);
      }
      if (auto eval = As<EvalStmt>(stmt)) call = As<Call>(eval->expr_);
      if (call) {
        for (const auto& arg : call->args_) CollectExprConstants(arg);
      }
      if (auto loop = As<ir::ForStmt>(stmt)) {
        CollectExprConstants(loop->start_);
        CollectExprConstants(loop->stop_);
        CollectExprConstants(loop->step_);
        for (const auto& iter_arg : loop->iter_args_) CollectExprConstants(iter_arg->initValue_);
        CollectStmtConstants(loop->body_);
      } else if (auto branch = As<ir::IfStmt>(stmt)) {
        CollectExprConstants(branch->condition_);
        CollectStmtConstants(branch->then_body_);
        if (branch->else_body_) CollectStmtConstants(*branch->else_body_);
      } else if (auto yield = As<ir::YieldStmt>(stmt)) {
        for (const auto& value : yield->value_) CollectExprConstants(value);
      } else if (auto ret = As<ReturnStmt>(stmt)) {
        for (const auto& value : ret->value_) CollectExprConstants(value);
      }
    }
  }

  void CollectTensorConstants(const TensorType& tensor) {
    INTERNAL_CHECK(!tensor.shape_.empty()) << "PTO tensor views require rank >= 1";
    for (const auto& dim : tensor.shape_) CollectExprConstants(dim);
    const auto tensor_view = ResolveLogicalTensorView(tensor);
    for (const auto& stride : tensor_view.strides) CollectExprConstants(stride);
    constants_.emplace(ConstantKey{1, DataType::INDEX.Code()}, DataType::INDEX);
  }

  std::string ConstantName(int64_t value, DataType dtype) const {
    std::string stem;
    if (value == 0) {
      stem = "c0";
    } else if (value < 0) {
      const uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
      stem = "cn" + std::to_string(magnitude);
    } else {
      stem = "c" + std::to_string(value);
    }
    return "%" + stem + "_" + DataTypeToMLIR(dtype);
  }

  void EmitConstants() {
    for (const auto& [key, dtype] : constants_) {
      const std::string name = ConstantName(key.first, dtype);
      used_names_.insert(name.substr(1));
      stream_ << Indent() << name << " = arith.constant " << key.first << " : " << DataTypeToMLIR(dtype)
              << "\n";
    }
    for (const auto& [key, info] : float_constants_) {
      (void)key;
      used_names_.insert(info.name.substr(1));
      stream_ << Indent() << info.name << " = arith.constant " << std::scientific
              << std::setprecision(std::numeric_limits<double>::max_digits10) << info.value << " : "
              << DataTypeToMLIR(info.dtype) << "\n";
    }
  }

  ConstIntPtr RequireConstInt(const ExprPtr& expr, const Span& span, const std::string& context) const {
    auto value = As<ConstInt>(expr);
    CHECK_SPAN(value, span) << "PTOIRPrinter Step-4 slice requires constant " << context;
    return value;
  }

  std::string ConstOperand(const ExprPtr& expr, const Span& span, const std::string& context) const {
    auto value = RequireConstInt(expr, span, context);
    return ConstantName(value->value_, value->dtype());
  }

  std::string FloatConstantName(const ConstFloat& value) const {
    const auto it = float_constants_.find(FloatConstantKey{FloatBits(value.value_), value.dtype().Code()});
    INTERNAL_CHECK_SPAN(it != float_constants_.end(), value.span_)
        << "Floating-point constant was not collected before PTO target printing";
    return it->second.name;
  }

  PrintedOperand ScalarOperand(const ExprPtr& expr, const Span& span) {
    auto scalar_type = As<ScalarType>(expr->GetType());
    INTERNAL_CHECK_SPAN(scalar_type, span) << "PTO scalar operand must have ScalarType";
    if (auto var = AsVarLike(expr)) {
      return PrintedOperand{VarName(var), DataTypeToMLIR(scalar_type->dtype_)};
    }
    if (auto value = As<ConstInt>(expr)) {
      return PrintedOperand{ConstantName(value->value_, value->dtype()), DataTypeToMLIR(scalar_type->dtype_)};
    }
    if (auto value = As<ConstFloat>(expr)) {
      return PrintedOperand{FloatConstantName(*value), DataTypeToMLIR(scalar_type->dtype_)};
    }

    auto binary = std::dynamic_pointer_cast<const ir::BinaryExpr>(expr);
    if (binary) {
      if (As<ir::Eq>(expr) || As<ir::Ne>(expr) || As<ir::Lt>(expr) || As<ir::Le>(expr) || As<ir::Gt>(expr) ||
          As<ir::Ge>(expr)) {
        auto lhs = ScalarOperand(binary->left_, span);
        auto rhs = ScalarOperand(binary->right_, span);
        CHECK_SPAN(lhs.type == rhs.type, span) << "PTO scalar comparison operands must have equal types";
        auto operand_type = As<ScalarType>(binary->left_->GetType());
        INTERNAL_CHECK_SPAN(operand_type, span) << "PTO scalar comparison operand must have ScalarType";
        const bool is_float = operand_type->dtype_.IsFloat();
        const bool is_unsigned = operand_type->dtype_.IsUnsignedInt();
        std::string predicate;
        if (As<ir::Eq>(expr)) predicate = is_float ? "oeq" : "eq";
        if (As<ir::Ne>(expr)) predicate = is_float ? "une" : "ne";
        if (As<ir::Lt>(expr)) predicate = is_float ? "olt" : (is_unsigned ? "ult" : "slt");
        if (As<ir::Le>(expr)) predicate = is_float ? "ole" : (is_unsigned ? "ule" : "sle");
        if (As<ir::Gt>(expr)) predicate = is_float ? "ogt" : (is_unsigned ? "ugt" : "sgt");
        if (As<ir::Ge>(expr)) predicate = is_float ? "oge" : (is_unsigned ? "uge" : "sge");
        const std::string result = FreshName("cmp");
        stream_ << Indent() << result << " = " << (is_float ? "arith.cmpf " : "arith.cmpi ") << predicate
                << ", " << lhs.value << ", " << rhs.value << " : " << lhs.type << "\n";
        return PrintedOperand{result, "i1"};
      }
      std::string int_op;
      std::string float_op;
      if (As<ir::Add>(expr)) {
        int_op = "arith.addi";
        float_op = "arith.addf";
      } else if (As<ir::Sub>(expr)) {
        int_op = "arith.subi";
        float_op = "arith.subf";
      } else if (As<ir::Mul>(expr)) {
        int_op = "arith.muli";
        float_op = "arith.mulf";
      } else if (As<ir::FloorDiv>(expr) || As<ir::FloatDiv>(expr)) {
        int_op = "arith.divsi";
        float_op = "arith.divf";
      } else if (As<ir::FloorMod>(expr)) {
        int_op = "arith.remsi";
        float_op = "arith.remf";
      } else if (As<ir::Min>(expr)) {
        int_op = scalar_type->dtype_.IsUnsignedInt() ? "arith.minui" : "arith.minsi";
        float_op = "arith.minimumf";
      } else if (As<ir::Max>(expr)) {
        int_op = scalar_type->dtype_.IsUnsignedInt() ? "arith.maxui" : "arith.maxsi";
        float_op = "arith.maximumf";
      }
      CHECK_SPAN(!int_op.empty(), span)
          << "PTOIRPrinter does not yet support scalar expression " << expr->TypeName();
      auto lhs = ScalarOperand(binary->left_, span);
      auto rhs = ScalarOperand(binary->right_, span);
      const std::string type = DataTypeToMLIR(scalar_type->dtype_);
      CHECK_SPAN(lhs.type == type && rhs.type == type, span)
          << "PTO scalar binary operands must match the result type";
      const std::string result = FreshName("scalar");
      stream_ << Indent() << result << " = " << (scalar_type->dtype_.IsFloat() ? float_op : int_op) << " "
              << lhs.value << ", " << rhs.value << " : " << type << "\n";
      return PrintedOperand{result, type};
    }

    if (auto cast = As<ir::Cast>(expr)) {
      auto source = ScalarOperand(cast->operand_, span);
      const auto source_type = As<ScalarType>(cast->operand_->GetType());
      INTERNAL_CHECK_SPAN(source_type, span) << "PTO scalar cast source must have ScalarType";
      const auto src_dtype = source_type->dtype_;
      const auto dst_dtype = scalar_type->dtype_;
      if (src_dtype == dst_dtype) return source;
      std::string op;
      if (src_dtype == DataType::INDEX || dst_dtype == DataType::INDEX) {
        op = "arith.index_cast";
      } else if (src_dtype.IsFloat() && dst_dtype.IsFloat()) {
        op = dst_dtype.GetBit() > src_dtype.GetBit() ? "arith.extf" : "arith.truncf";
      } else if (!src_dtype.IsFloat() && !dst_dtype.IsFloat()) {
        op = dst_dtype.GetBit() > src_dtype.GetBit() ? "arith.extsi" : "arith.trunci";
      } else {
        op = src_dtype.IsFloat() ? "arith.fptosi" : "arith.sitofp";
      }
      const std::string result = FreshName("cast");
      const std::string type = DataTypeToMLIR(dst_dtype);
      stream_ << Indent() << result << " = " << op << " " << source.value << " : " << source.type << " to "
              << type << "\n";
      return PrintedOperand{result, type};
    }

    CHECK_SPAN(false, span) << "PTOIRPrinter does not yet support scalar operand " << expr->TypeName();
    return {};
  }

  std::string IndexOperand(const ExprPtr& expr, const Span& span, const std::string& context) {
    auto operand = ScalarOperand(expr, span);
    if (operand.type == "index") return operand.value;
    auto scalar_type = As<ScalarType>(expr->GetType());
    CHECK_SPAN(scalar_type && scalar_type->dtype_.IsInt(), span)
        << context << " must be integer or index typed";
    const std::string result = FreshName("index");
    stream_ << Indent() << result << " = arith.index_cast " << operand.value << " : " << operand.type
            << " to index\n";
    return result;
  }

  std::string FreshName(const std::string& hint) {
    std::string base = SanitizeName(hint);
    if (used_names_.insert(base).second) return "%" + base;
    while (true) {
      std::string candidate = base + "_" + std::to_string(next_temp_++);
      if (used_names_.insert(candidate).second) return "%" + candidate;
    }
  }

  std::string Indent() const { return std::string(indent_, ' '); }

  std::string VarName(const VarPtr& var) {
    auto it = var_names_.find(var.get());
    if (it != var_names_.end()) return it->second;
    auto name = FreshName(var->name_hint_);
    var_names_[var.get()] = name;
    return name;
  }

  void EmitTensorView(const VarPtr& tensor_var, const VarPtr& base_ptr) {
    auto tensor = AsTensorTypeLike(tensor_var->GetType());
    INTERNAL_CHECK_SPAN(tensor && !tensor->shape_.empty(), tensor_var->span_)
        << "PTO tensor view result must have rank >= 1 TensorType";

    const auto tensor_view = ResolveLogicalTensorView(*tensor);
    CHECK_SPAN(tensor_view.layout != TensorLayout::NZ, tensor_var->span_)
        << "PTO logical tensor views support only ND or DN layout";

    INTERNAL_CHECK_SPAN(tensor_view.strides.size() == tensor->shape_.size(), tensor_var->span_)
        << "PTO tensor view stride rank must match shape rank";

    std::vector<std::string> shape_operands;
    std::vector<std::string> stride_operands;
    shape_operands.reserve(tensor->shape_.size());
    stride_operands.reserve(tensor_view.strides.size());
    for (const auto& dim : tensor->shape_) {
      shape_operands.push_back(IndexOperand(dim, tensor_var->span_, "Tensor shape"));
    }
    for (const auto& stride : tensor_view.strides) {
      stride_operands.push_back(IndexOperand(stride, tensor_var->span_, "Tensor stride"));
    }

    const std::string view = FreshName(tensor_var->name_hint_ + "_view");
    tensor_views_[tensor_var.get()] = view;
    stream_ << Indent() << view << " = pto.make_tensor_view " << VarName(base_ptr) << ", shape = [";
    for (size_t i = 0; i < shape_operands.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << shape_operands[i];
    }
    stream_ << "], strides = [";
    for (size_t i = 0; i < stride_operands.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << stride_operands[i];
    }
    stream_ << "] {layout = #pto.layout<" << (tensor_view.layout == TensorLayout::DN ? "dn" : "nd")
            << ">}: " << TensorViewTypeString(*tensor) << "\n";
  }

  void EmitTensorViews(const FunctionPtr& function) {
    for (const auto& param : function->params_) {
      if (AsTensorTypeLike(param->GetType())) EmitTensorView(param, param);
    }
  }

  void PrintStmt(const StmtPtr& stmt) {
    if (auto assign = As<AssignStmt>(stmt)) {
      PrintAssign(assign);
      return;
    }
    if (auto eval = As<EvalStmt>(stmt)) {
      auto call = As<Call>(eval->expr_);
      if (call && IsOp(call, "tensor.write")) {
        PrintTensorWrite(std::nullopt, call);
        return;
      }
      INTERNAL_CHECK_SPAN(call && call->op_->name_.rfind("pto.", 0) == 0, eval->span_)
          << "PTOIRPrinter target EvalStmt must contain a PTO Call";
      PrintTargetCall(call);
      return;
    }
    if (auto loop = As<ir::ForStmt>(stmt)) {
      PrintFor(loop);
      return;
    }
    if (auto branch = As<ir::IfStmt>(stmt)) {
      PrintIf(branch);
      return;
    }
    if (auto yield = As<ir::YieldStmt>(stmt)) {
      PrintYield(yield);
      return;
    }
    if (As<ReturnStmt>(stmt)) return;
    CHECK_SPAN(false, stmt->span_) << "PTOIRPrinter does not support structured target statement "
                                   << stmt->TypeName();
  }

  void PrintBody(const StmtPtr& body) {
    for (const auto& stmt : transform_utils::FlattenToStmts(body)) PrintStmt(stmt);
  }

  void BindTensorAlias(const VarPtr& alias, const ExprPtr& source_expr, const Span& span) {
    auto source = AsVarLike(source_expr);
    INTERNAL_CHECK_SPAN(source && AsTensorTypeLike(source->GetType()), span)
        << "Structured tensor alias must have a tensor Var initial value";
    var_names_[alias.get()] = VarName(source);
    auto view = tensor_views_.find(source.get());
    INTERNAL_CHECK_SPAN(view != tensor_views_.end(), span)
        << "Structured tensor alias source has no materialized tensor view";
    tensor_views_[alias.get()] = view->second;
  }

  void PrintFor(const ir::ForStmtPtr& loop) {
    INTERNAL_CHECK_SPAN(loop->iter_args_.size() == loop->return_vars_.size(), loop->span_)
        << "Target ForStmt carry arity mismatch";
    const std::string start = IndexOperand(loop->start_, loop->span_, "for start");
    const std::string stop = IndexOperand(loop->stop_, loop->span_, "for stop");
    const std::string step = IndexOperand(loop->step_, loop->span_, "for step");
    const std::string loop_var = FreshName(loop->loop_var_->name_hint_);
    var_names_[loop->loop_var_.get()] = loop_var;

    std::vector<std::string> scalar_names;
    std::vector<std::string> scalar_initializers;
    std::vector<std::string> scalar_types;
    std::vector<std::string> scalar_results;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& iter_arg = loop->iter_args_[i];
      const auto& return_var = loop->return_vars_[i];
      if (auto scalar = As<ScalarType>(iter_arg->GetType())) {
        auto init = ScalarOperand(iter_arg->initValue_, loop->span_);
        const std::string iter_name = FreshName(iter_arg->name_hint_);
        const std::string result_name = FreshName(return_var->name_hint_);
        var_names_[iter_arg.get()] = iter_name;
        var_names_[return_var.get()] = result_name;
        scalar_names.push_back(iter_name);
        scalar_initializers.push_back(init.value);
        scalar_types.push_back(DataTypeToMLIR(scalar->dtype_));
        scalar_results.push_back(result_name);
      } else if (AsTensorTypeLike(iter_arg->GetType())) {
        BindTensorAlias(iter_arg, iter_arg->initValue_, loop->span_);
        BindTensorAlias(return_var, iter_arg->initValue_, loop->span_);
      } else {
        CHECK_SPAN(false, loop->span_)
            << "Target ForStmt supports only scalar and tensor carries after Tile bufferization";
      }
    }

    stream_ << Indent();
    if (!scalar_results.empty()) {
      for (size_t i = 0; i < scalar_results.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << scalar_results[i];
      }
      stream_ << " = ";
    }
    stream_ << "scf.for " << loop_var << " = " << start << " to " << stop << " step " << step;
    if (!scalar_names.empty()) {
      stream_ << " iter_args(";
      for (size_t i = 0; i < scalar_names.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << scalar_names[i] << " = " << scalar_initializers[i];
      }
      stream_ << ") -> (";
      for (size_t i = 0; i < scalar_types.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << scalar_types[i];
      }
      stream_ << ")";
    }
    stream_ << " {\n";
    indent_ += 2;
    yield_types_.push_back(scalar_types);
    PrintBody(loop->body_);
    yield_types_.pop_back();
    indent_ -= 2;
    stream_ << Indent() << "}\n";
  }

  void PrintIf(const ir::IfStmtPtr& branch) {
    const auto condition = ScalarOperand(branch->condition_, branch->span_);
    CHECK_SPAN(condition.type == "i1", branch->span_) << "Target IfStmt condition must be i1";
    std::vector<std::string> scalar_results;
    std::vector<std::string> scalar_types;
    for (const auto& result : branch->return_vars_) {
      if (auto scalar = As<ScalarType>(result->GetType())) {
        auto name = FreshName(result->name_hint_);
        var_names_[result.get()] = name;
        scalar_results.push_back(name);
        scalar_types.push_back(DataTypeToMLIR(scalar->dtype_));
      } else {
        CHECK_SPAN(AsTensorTypeLike(result->GetType()), branch->span_)
            << "Target IfStmt supports only scalar and tensor results after Tile bufferization";
      }
    }
    stream_ << Indent();
    if (!scalar_results.empty()) {
      for (size_t i = 0; i < scalar_results.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << scalar_results[i];
      }
      stream_ << " = ";
    }
    stream_ << "scf.if " << condition.value;
    if (!scalar_types.empty()) {
      stream_ << " -> (";
      for (size_t i = 0; i < scalar_types.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << scalar_types[i];
      }
      stream_ << ")";
    }
    stream_ << " {\n";
    indent_ += 2;
    yield_types_.push_back(scalar_types);
    PrintBody(branch->then_body_);
    yield_types_.pop_back();
    indent_ -= 2;
    if (branch->else_body_) {
      stream_ << Indent() << "} else {\n";
      indent_ += 2;
      yield_types_.push_back(scalar_types);
      PrintBody(*branch->else_body_);
      yield_types_.pop_back();
      indent_ -= 2;
    }
    stream_ << Indent() << "}\n";
  }

  void PrintYield(const ir::YieldStmtPtr& yield) {
    INTERNAL_CHECK_SPAN(!yield_types_.empty(), yield->span_)
        << "YieldStmt must be nested in a target control-flow region";
    const auto& expected_types = yield_types_.back();
    std::vector<PrintedOperand> values;
    for (const auto& value : yield->value_) {
      if (IsA<ScalarType>(value->GetType())) values.push_back(ScalarOperand(value, yield->span_));
    }
    INTERNAL_CHECK_SPAN(values.size() == expected_types.size(), yield->span_)
        << "Target scalar yield count does not match structured result count";
    if (values.empty()) return;
    stream_ << Indent() << "scf.yield ";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << values[i].value;
    }
    stream_ << " : ";
    for (size_t i = 0; i < expected_types.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << expected_types[i];
    }
    stream_ << "\n";
  }

  void PrintAssign(const AssignStmtPtr& assign) {
    if (auto call = As<Call>(assign->value_)) {
      if (IsOp(call, "pto.alloc_tile")) {
        PrintAllocation(assign, call);
        return;
      }
      if (IsOp(call, "pto.subview")) {
        PrintSubview(assign, call);
        return;
      }
      if (IsOp(call, "tensor.view")) {
        INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
            << "tensor.view target assignment requires a source tensor";
        auto source = AsVarLike(call->args_[0]);
        INTERNAL_CHECK_SPAN(
            source && AsTensorTypeLike(source->GetType()) && AsTensorTypeLike(assign->var_->GetType()),
            call->span_)
            << "tensor.view target assignment requires Tensor source and result";
        // The view has its own logical metadata SSA but aliases the source raw
        // pointer.  Keeping both bindings explicit avoids recovering aliasing
        // by inspecting downstream uses in codegen.
        var_names_[assign->var_.get()] = VarName(source);
        EmitTensorView(assign->var_, assign->var_);
        return;
      }
      if (IsOp(call, "tensor.read")) {
        PrintTensorRead(assign, call);
        return;
      }
      if (IsOp(call, "tensor.write")) {
        PrintTensorWrite(assign->var_, call);
        return;
      }
      CHECK_SPAN(false, call->span_) << "PTOIRPrinter target scalar Call assignment is not yet materialized";
      return;
    }
    if (IsA<ScalarType>(assign->var_->GetType())) {
      auto operand = ScalarOperand(assign->value_, assign->span_);
      var_names_[assign->var_.get()] = operand.value;
      return;
    }
    auto source = AsVarLike(assign->value_);
    auto tensor = source ? AsTensorTypeLike(source->GetType()) : nullptr;
    CHECK_SPAN(source && tensor, assign->span_)
        << "PTOIRPrinter target non-Call assignment must be a Tensor alias";
    var_names_[assign->var_.get()] = VarName(source);
    auto view = tensor_views_.find(source.get());
    INTERNAL_CHECK_SPAN(view != tensor_views_.end(), assign->span_)
        << "Tensor alias source has no materialized tensor view";
    tensor_views_[assign->var_.get()] = view->second;
  }

  std::string FlatTensorOffset(const ir::MakeTuplePtr& indices, const std::vector<ExprPtr>& shape,
                               const Span& span) {
    INTERNAL_CHECK_SPAN(indices && indices->elements_.size() == shape.size(), span)
        << "Tensor scalar access index rank must match tensor rank";
    auto offset = IndexOperand(indices->elements_[0], span, "tensor index");
    for (size_t i = 1; i < indices->elements_.size(); ++i) {
      auto extent = IndexOperand(shape[i], span, "tensor extent");
      auto scaled = FreshName("flat_offset_mul");
      stream_ << Indent() << scaled << " = arith.muli " << offset << ", " << extent << " : index\n";
      auto index = IndexOperand(indices->elements_[i], span, "tensor index");
      auto next = FreshName("flat_offset");
      stream_ << Indent() << next << " = arith.addi " << scaled << ", " << index << " : index\n";
      offset = next;
    }
    return offset;
  }

  void PrintTensorRead(const AssignStmtPtr& assign, const CallPtr& call) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 2, call->span_)
        << "tensor.read target assignment requires tensor and indices";
    auto tensor = AsVarLike(call->args_[0]);
    auto tensor_type = tensor ? AsTensorTypeLike(tensor->GetType()) : nullptr;
    auto indices = As<MakeTuple>(call->args_[1]);
    auto result_type = As<ScalarType>(assign->var_->GetType());
    INTERNAL_CHECK_SPAN(tensor_type && indices && result_type, call->span_)
        << "Malformed tensor.read target assignment";
    auto offset = FlatTensorOffset(indices, tensor_type->shape_, call->span_);
    auto result = VarName(assign->var_);
    stream_ << Indent() << result << " = pto.load_scalar " << VarName(tensor) << "[" << offset
            << "] : !pto.ptr<" << DataTypeToMLIR(tensor_type->dtype_) << "> -> "
            << DataTypeToMLIR(result_type->dtype_) << "\n";
  }

  void PrintTensorWrite(const std::optional<VarPtr>& result_var, const CallPtr& call) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
        << "tensor.write target assignment requires tensor, indices, and value";
    auto tensor = AsVarLike(call->args_[0]);
    auto tensor_type = tensor ? AsTensorTypeLike(tensor->GetType()) : nullptr;
    auto indices = As<MakeTuple>(call->args_[1]);
    INTERNAL_CHECK_SPAN(
        tensor_type && indices && (!result_var.has_value() || AsTensorTypeLike((*result_var)->GetType())),
        call->span_)
        << "Malformed tensor.write target assignment";
    auto value = ScalarOperand(call->args_[2], call->span_);
    auto offset = FlatTensorOffset(indices, tensor_type->shape_, call->span_);
    stream_ << Indent() << "pto.store_scalar " << value.value << ", " << VarName(tensor) << "[" << offset
            << "] : !pto.ptr<" << DataTypeToMLIR(tensor_type->dtype_) << ">, " << value.type << "\n";
    if (result_var) {
      var_names_[result_var->get()] = VarName(tensor);
      auto view = tensor_views_.find(tensor.get());
      if (view != tensor_views_.end()) tensor_views_[result_var->get()] = view->second;
    }
  }

  void PrintAllocation(const AssignStmtPtr& assign, const CallPtr& call) {
    auto type = As<PTOTileBufType>(assign->var_->GetType());
    INTERNAL_CHECK_SPAN(type && structural_equal(type, call->GetType()), call->span_)
        << "pto.alloc_tile assignment and result types must match";
    INTERNAL_CHECK_SPAN(call->args_.size() == 2 || call->args_.size() == 3, call->span_)
        << "pto.alloc_tile requires valid-row and valid-col operands";
    const std::string valid_row =
        IndexOperand(call->args_[call->args_.size() - 2], call->span_, "allocation valid row");
    const std::string valid_col = IndexOperand(call->args_.back(), call->span_, "allocation valid column");
    stream_ << Indent() << VarName(assign->var_) << " = pto.alloc_tile";
    if (emit_tile_addr_ && call->args_.size() == 3) {
      stream_ << " addr = " << ConstOperand(call->args_[0], call->span_, "allocation address");
    }
    stream_ << " valid_row = " << valid_row << " valid_col = " << valid_col << " : "
            << TileBufferTypeString(*type) << "\n";
  }

  std::vector<std::string> TupleOperands(const ExprPtr& expr, const Span& span, const std::string& context) {
    auto tuple = As<MakeTuple>(expr);
    CHECK_SPAN(tuple && !tuple->elements_.empty(), span)
        << "PTOIRPrinter requires a non-empty " << context << " tuple";
    std::vector<std::string> operands;
    for (const auto& element : tuple->elements_) {
      operands.push_back(IndexOperand(element, span, context));
    }
    return operands;
  }

  void PrintSubview(const AssignStmtPtr& assign, const CallPtr& call) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 4, call->span_)
        << "pto.subview requires source, shape, offset, and valid-shape operands";
    auto source = AsVarLike(call->args_[0]);
    auto source_type = source ? As<PTOTileBufType>(source->GetType()) : nullptr;
    auto result_type = As<PTOTileBufType>(assign->var_->GetType());
    INTERNAL_CHECK_SPAN(source_type && result_type && structural_equal(result_type, call->GetType()),
                        call->span_)
        << "Malformed pto.subview source or result type";
    auto offsets = TupleOperands(call->args_[2], call->span_, "subview offset");
    auto valid = TupleOperands(call->args_[3], call->span_, "subview valid shape");
    INTERNAL_CHECK_SPAN(offsets.size() == 2 && valid.size() == 2, call->span_)
        << "pto.subview offset and valid shape must be rank-2";
    stream_ << Indent() << VarName(assign->var_) << " = pto.subview " << VarName(source) << "[" << offsets[0]
            << ", " << offsets[1] << "] sizes [" << result_type->rows_ << ", " << result_type->cols_
            << "] valid [" << valid[0] << ", " << valid[1] << "] : " << TileBufferTypeString(*source_type)
            << " -> " << TileBufferTypeString(*result_type) << "\n";
  }

  std::string TensorViewFor(const VarPtr& tensor, const Span& span) const {
    auto it = tensor_views_.find(tensor.get());
    INTERNAL_CHECK_SPAN(it != tensor_views_.end(), span)
        << "PTO target tensor operand '" << tensor->name_hint_ << "' has no tensor view";
    return it->second;
  }

  std::string EmitPartitionView(const VarPtr& tensor, const ExprPtr& offsets_expr, const ExprPtr& sizes_expr,
                                const Span& span) {
    auto tensor_type = AsTensorTypeLike(tensor->GetType());
    INTERNAL_CHECK_SPAN(tensor_type, span) << "PTO partition operand must have TensorType";
    auto static_sizes = As<MakeTuple>(sizes_expr);
    INTERNAL_CHECK_SPAN(static_sizes, span) << "PTO partition size must be a MakeTuple";
    auto offsets = TupleOperands(offsets_expr, span, "partition offset");
    auto sizes = TupleOperands(sizes_expr, span, "partition size");
    INTERNAL_CHECK_SPAN(offsets.size() == sizes.size(), span)
        << "PTO partition offsets and sizes must have the same rank";
    const std::string partition_type = PartitionTypeString(*static_sizes, tensor_type->dtype_);
    const std::string partition = FreshName(tensor->name_hint_ + "_pview");
    stream_ << Indent() << partition << " = pto.partition_view " << TensorViewFor(tensor, span)
            << ", offsets = [";
    for (size_t i = 0; i < offsets.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << offsets[i];
    }
    stream_ << "], sizes = [";
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << sizes[i];
    }
    stream_ << "] : " << TensorViewTypeString(*tensor_type) << " -> " << partition_type << "\n";
    return partition + " : " + partition_type;
  }

  void PrintTargetCall(const CallPtr& call) {
    if (IsOp(call, "pto.tload")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 4, call->span_) << "pto.tload requires four operands";
      auto source = AsVarLike(call->args_[0]);
      auto output = AsVarLike(call->args_[3]);
      auto output_type = output ? As<PTOTileBufType>(output->GetType()) : nullptr;
      INTERNAL_CHECK_SPAN(source && output_type, call->span_) << "Malformed pto.tload operands";
      const std::string partition = EmitPartitionView(source, call->args_[1], call->args_[2], call->span_);
      stream_ << Indent() << "pto.tload ins(" << partition << ") outs(" << VarName(output) << " : "
              << TileBufferTypeString(*output_type) << ")\n";
      return;
    }
    if (IsOp(call, "pto.tstore")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 4, call->span_) << "pto.tstore requires four operands";
      auto input = AsVarLike(call->args_[0]);
      auto input_type = input ? As<PTOTileBufType>(input->GetType()) : nullptr;
      auto destination = AsVarLike(call->args_[3]);
      INTERNAL_CHECK_SPAN(input_type && destination, call->span_) << "Malformed pto.tstore operands";
      const std::string partition =
          EmitPartitionView(destination, call->args_[1], call->args_[2], call->span_);
      stream_ << Indent() << "pto.tstore ins(" << VarName(input) << " : " << TileBufferTypeString(*input_type)
              << ") outs(" << partition << ")";
      const int atomic = call->GetKwarg<int>("atomic", static_cast<int>(AtomicType::kNone));
      INTERNAL_CHECK_SPAN(
          atomic == static_cast<int>(AtomicType::kNone) || atomic == static_cast<int>(AtomicType::kAdd),
          call->span_)
          << "pto.tstore atomic kwarg must encode AtomicType::kNone or AtomicType::kAdd";
      if (atomic == static_cast<int>(AtomicType::kAdd)) {
        auto destination_type = AsTensorTypeLike(destination->GetType());
        CHECK_SPAN(destination_type, call->span_) << "pto.tstore destination must have TensorType";
        CHECK_SPAN(destination_type->dtype_ != DataType::BF16 || target_arch_ == "a2a3", call->span_)
            << "tile.store with atomic=AtomicType.Add into a bf16 global tensor is not supported on the '"
            << target_arch_
            << "' backend; bf16 atomic-add requires the Ascend910B (A2/A3) profile. Accumulate into an "
               "fp32 tensor and cast to bf16 after the reduction instead.";
        stream_ << " {atomicType = #pto<atomic_type atomic_add>}";
      }
      stream_ << "\n";
      return;
    }
    PrintDestinationPassingCall(call);
  }

  void PrintDestinationPassingCall(const CallPtr& call) {
    const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
    const auto& spec = entry.GetPTOOpSpec();
    INTERNAL_CHECK_SPAN(spec.has_value(), call->span_)
        << "PTO target operation has no operand/effect schema: '" << call->op_->name_ << "'";
    auto segments = spec->ResolveOperandSegments(call->args_.size());
    INTERNAL_CHECK_SPAN(segments.has_value(), call->span_)
        << "PTO target operation has invalid operand count: '" << call->op_->name_ << "'";

    std::vector<PrintedOperand> inputs;
    std::vector<VarPtr> outputs;
    size_t arg_index = 0;
    for (size_t group_index = 0; group_index < spec->operand_groups.size(); ++group_index) {
      const auto& group = spec->operand_groups[group_index];
      for (size_t i = 0; i < (*segments)[group_index]; ++i, ++arg_index) {
        const auto& arg = call->args_[arg_index];
        if (group.role == PTOOperandRole::Output) {
          auto handle = AsVarLike(arg);
          INTERNAL_CHECK_SPAN(handle && IsA<PTOTileBufType>(handle->GetType()), call->span_)
              << "PTO arithmetic output must be an explicit tile-buffer handle";
          outputs.push_back(handle);
          continue;
        }
        INTERNAL_CHECK_SPAN(group.role == PTOOperandRole::Input, call->span_)
            << "Generic PTO arithmetic printer does not accept metadata operand groups";
        if (group.type_constraint == ir::PTOOperandTypeConstraint::TileBuffer) {
          auto handle = AsVarLike(arg);
          auto type = handle ? As<PTOTileBufType>(handle->GetType()) : nullptr;
          INTERNAL_CHECK_SPAN(type, call->span_)
              << "PTO arithmetic tile input must be an explicit tile-buffer handle";
          inputs.push_back(PrintedOperand{VarName(handle), TileBufferTypeString(*type)});
        } else if (group.type_constraint == ir::PTOOperandTypeConstraint::Scalar) {
          inputs.push_back(ScalarOperand(arg, call->span_));
        } else {
          INTERNAL_CHECK_SPAN(false, call->span_)
              << "Generic PTO arithmetic printer requires typed TileBuffer or Scalar inputs";
        }
      }
    }
    INTERNAL_CHECK_SPAN(!inputs.empty() && outputs.size() == 1, call->span_)
        << "PTO arithmetic operation must have inputs and exactly one output";

    stream_ << Indent() << call->op_->name_ << " ins(";
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << inputs[i].value;
    }
    stream_ << " : ";
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << inputs[i].type;
    }
    stream_ << ") outs(" << VarName(outputs[0]) << " : "
            << TileBufferTypeString(*As<PTOTileBufType>(outputs[0]->GetType())) << ")\n";
  }

  std::string target_arch_;
  bool emit_tile_addr_;
  std::ostringstream stream_;
  std::unordered_map<const Var*, std::string> var_names_;
  std::unordered_map<const Var*, std::string> tensor_views_;
  std::map<ConstantKey, DataType> constants_;
  std::map<FloatConstantKey, FloatConstantInfo> float_constants_;
  std::set<std::string> used_names_;
  size_t next_temp_ = 0;
  size_t indent_ = 2;
  std::vector<std::vector<std::string>> yield_types_;
};

}  // namespace

PTOIRPrinter::PTOIRPrinter(const backend::Backend* backend) : backend_(backend) {
  CHECK(backend_ && backend_->GetHandler()) << "PTOIRPrinter requires a backend with a BackendHandler";
}

std::string PTOIRPrinter::Generate(const ProgramPtr& program, bool emit_tile_addr) {
  return FunctionPrinter(backend_->GetHandler()->GetPtoTargetArch(), emit_tile_addr).PrintProgram(program);
}

std::string PTOIRPrinter::GenerateFunction(const ir::FunctionPtr& function, bool emit_tile_addr) {
  return FunctionPrinter(backend_->GetHandler()->GetPtoTargetArch(), emit_tile_addr)
      .PrintSingleFunction(function);
}

}  // namespace codegen
}  // namespace pypto
