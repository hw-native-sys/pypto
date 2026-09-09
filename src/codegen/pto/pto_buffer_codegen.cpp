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

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace codegen {
namespace {

using ir::As;

// BufferType is already a physical descriptor. This initial emitter supports
// dense Vec fp16/fp32 only; no logical shape, layout, or packing is inferred.
std::string BufferTypeString(const ir::BufferTypePtr& type, const ir::Span& span) {
  CHECK_SPAN(type->shape_.size() == 1 || type->shape_.size() == 2, span)
      << "Direct Buffer IR codegen supports only rank-1 or rank-2 buffers";
  CHECK_SPAN(type->memory_space_ == ir::MemorySpace::Vec &&
                 (type->dtype_ == DataType::FP16 || type->dtype_ == DataType::FP32) &&
                 type->blayout_ == ir::TileLayout::row_major && type->slayout_ == ir::TileLayout::none_box &&
                 type->fractal_ == 512 && type->pad_ == ir::PadValue::null &&
                 type->compact_ == ir::CompactMode::null,
             span)
      << "Direct Buffer IR codegen currently requires dense row-major Vec FP16/FP32 buffers "
         "with fractal=512, no padding, and no compact mode";
  const bool vector = type->shape_.size() == 1;
  const int64_t rows = vector ? 1 : type->shape_[0];
  const int64_t cols = type->shape_.back();
  const int64_t valid_rows = vector ? 1 : type->valid_shape_[0];
  const int64_t valid_cols = type->valid_shape_.back();
  return FormatTileBufTypeString("vec", DataTypeToMLIR(type->dtype_), rows, cols, type->blayout_,
                                 type->slayout_, type->fractal_, type->pad_, type->compact_, valid_rows,
                                 valid_cols, valid_rows == -1, valid_cols == -1);
}

class BufferFunctionDetector : public ir::IRVisitor {
 public:
  bool found = false;

  void VisitFunction(const ir::FunctionPtr& function) override {
    for (const auto& type : function->return_types_) VisitType(type);
    IRVisitor::VisitFunction(function);
  }

  void VisitExpr(const ir::ExprPtr& expr) override {
    if (!expr) return;
    VisitType(expr->GetType());
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt(const ir::StmtPtr& stmt) override {
    // The default visitor omits For attrs and some scope kinds' attrs. A
    // buffer referenced only there must still select representation validation
    // rather than silently disappearing through the legacy emitter.
    if (auto scope = As<ir::ScopeStmt>(stmt)) VisitAttrs(scope->attrs_);
    if (auto loop = As<ir::ForStmt>(stmt)) VisitAttrs(loop->attrs_);
    IRVisitor::VisitStmt(stmt);
  }

 protected:
  void VisitVarLike_(const ir::VarPtr&) override {}

  [[nodiscard]] bool ShouldVisitScopeAttr(const std::string&) const override { return false; }

  void VisitExpr_(const ir::CallPtr& call) override {
    VisitAttrs(call->kwargs_);
    if (!call->op_) {
      found = true;
      IRVisitor::VisitExpr_(call);
      return;
    }
    const auto& registry = ir::OpRegistry::GetInstance();
    const auto& name = call->op_->name_;
    if (!As<ir::GlobalVar>(call->op_)) {
      // Include malformed buffer-family calls so the representation verifier
      // diagnoses them instead of falling through the legacy Tile emitter.
      found |= name.rfind("buffer.", 0) == 0 ||
               (registry.IsRegistered(name) && registry.GetEntry(name).GetIRStage() == ir::OpIRStage::Buffer);
    }
    IRVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const ir::SubmitPtr& submit) override {
    VisitAttrs(submit->kwargs_);
    IRVisitor::VisitExpr_(submit);
  }

 private:
  void VisitType(const ir::TypePtr& type) {
    if (!type || !visited_types_.insert(type.get()).second) return;
    found |= As<ir::BufferType>(type) != nullptr || As<ir::MultiBufferType>(type) != nullptr;
    if (auto tuple = As<ir::TupleType>(type)) {
      for (const auto& element : tuple->types_) VisitType(element);
    } else {
      ir::var_collectors::VisitTypeExprFields(*this, type);
      if (auto tensor = ir::AsTensorTypeLike(type); tensor && tensor->memref_ && *tensor->memref_) {
        const auto& memref = *tensor->memref_;
        if (visited_memrefs_.insert(memref.get()).second) {
          VisitExpr(memref->base_);
          VisitExpr(memref->byte_offset_);
        }
      }
    }
  }

  void VisitAttrs(const std::vector<std::pair<std::string, std::any>>& attrs) {
    for (const auto& [key, value] : attrs) {
      ir::ForEachAttrExpr(value, [this](const ir::ExprPtr& expr) { VisitExpr(expr); });
    }
  }

  std::unordered_set<const ir::Type*> visited_types_;
  std::unordered_set<const ir::MemRef*> visited_memrefs_;
};

// One additional walk rejects unsupported native recipes before emission. The
// representation verifier separately owns SSA, dominance, and op contracts.
class BufferEmissionPreflight : public ir::IRVisitor {
 public:
  void VisitExpr(const ir::ExprPtr& expr) override {
    if (!expr) return;
    // The inherited visitor has default traversal-only handlers as well as
    // native scalar emitters. Reject those traversal-only kinds explicitly.
    using ir::ObjectKind;
    switch (expr->GetKind()) {
      case ObjectKind::Var:
      case ObjectKind::IterArg:
      case ObjectKind::ConstInt:
      case ObjectKind::ConstFloat:
      case ObjectKind::ConstBool:
      case ObjectKind::Call:
      case ObjectKind::MakeTuple:
      case ObjectKind::Add:
      case ObjectKind::Sub:
      case ObjectKind::Mul:
      case ObjectKind::FloorDiv:
      case ObjectKind::FloorMod:
      case ObjectKind::FloatDiv:
      case ObjectKind::Min:
      case ObjectKind::Max:
      case ObjectKind::Eq:
      case ObjectKind::Ne:
      case ObjectKind::Lt:
      case ObjectKind::Le:
      case ObjectKind::Gt:
      case ObjectKind::Ge:
      case ObjectKind::Cast:
      case ObjectKind::And:
      case ObjectKind::Or:
      case ObjectKind::Xor:
      case ObjectKind::BitAnd:
      case ObjectKind::BitOr:
      case ObjectKind::BitXor:
      case ObjectKind::BitShiftLeft:
      case ObjectKind::BitShiftRight:
      case ObjectKind::Not:
      case ObjectKind::Neg:
      case ObjectKind::Abs:
      case ObjectKind::BitNot:
        break;
      default:
        CHECK_SPAN(false, expr->span_)
            << "Expression '" << expr->TypeName() << "' is not supported by direct Buffer IR codegen";
    }
    if (auto type = As<ir::BufferType>(expr->GetType())) {
      if (descriptors_.insert(type.get()).second) BufferTypeString(type, expr->span_);
    }
    CHECK_SPAN(!As<ir::MultiBufferType>(expr->GetType()), expr->span_)
        << "Direct Buffer IR codegen does not yet support multi-buffer allocations";
    CHECK_SPAN(!As<ir::Submit>(expr) && !As<ir::TupleGetItemExpr>(expr), expr->span_)
        << "Direct Buffer IR codegen does not yet support task submissions or tuple projections";
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt(const ir::StmtPtr& stmt) override {
    if (!stmt) return;
    CHECK_SPAN(!returned_, stmt->span_) << "Direct Buffer IR codegen requires return to end the function";
    const bool region = As<ir::IfStmt>(stmt) || As<ir::ForStmt>(stmt) || As<ir::WhileStmt>(stmt);
    CHECK_SPAN(region || As<ir::SeqStmts>(stmt) || As<ir::AssignStmt>(stmt) || As<ir::EvalStmt>(stmt) ||
                   As<ir::YieldStmt>(stmt) || As<ir::ReturnStmt>(stmt),
               stmt->span_)
        << "Statement '" << stmt->TypeName() << "' is not supported by direct Buffer IR codegen";
    if (auto ret = As<ir::ReturnStmt>(stmt)) {
      CHECK_SPAN(depth_ == 0 && ret->value_.empty(), ret->span_)
          << "Direct Buffer IR codegen supports only a final empty function return";
      returned_ = true;
    }
    if (auto assign = As<ir::AssignStmt>(stmt)) {
      CHECK_SPAN(As<ir::BufferType>(assign->var_->GetType()) || As<ir::ScalarType>(assign->var_->GetType()),
                 assign->span_)
          << "Direct Buffer IR codegen supports only scalar or buffer assignments";
    }
    if (auto loop = As<ir::ForStmt>(stmt)) {
      CHECK_SPAN(ir::GetScalarDtype(loop->loop_var_) == DataType::INDEX, loop->span_)
          << "Direct Buffer IR codegen requires an INDEX for-loop induction variable";
      for (const auto& bound : {loop->start_, loop->stop_, loop->step_}) {
        const auto dtype = ir::GetScalarDtype(bound);
        CHECK_SPAN(dtype == DataType::INDEX || dtype.IsSignedInt(), bound->span_)
            << "Direct Buffer IR codegen requires INDEX or signed integer for-loop bounds";
      }
    }
    if (region) ++depth_;
    IRVisitor::VisitStmt(stmt);
    if (region) --depth_;
  }

 protected:
  void VisitBinaryExpr_(const ir::BinaryExprPtr& expr) override {
    CheckSignedArithmetic(expr);
    CheckSignedArithmetic(expr->left_);
    CheckSignedArithmetic(expr->right_);
    IRVisitor::VisitBinaryExpr_(expr);
  }

  void VisitUnaryExpr_(const ir::UnaryExprPtr& expr) override {
    if (As<ir::Cast>(expr)) {
      const auto source = ir::GetScalarDtype(expr->operand_);
      const auto target = ir::GetScalarDtype(expr);
      // The inherited emitter supports index -> unsigned and same-width
      // integer bitcasts. Other unsigned casts need a separate native recipe.
      CHECK_SPAN((!source.IsUnsignedInt() && !target.IsUnsignedInt()) || source == DataType::INDEX ||
                     source == target ||
                     (!source.IsFloat() && target != DataType::INDEX && !target.IsFloat() &&
                      source.GetBit() == target.GetBit()),
                 expr->span_)
          << "Unsigned scalar cast is not supported by direct Buffer IR codegen; "
             "pass unsigned address/valid operands directly to buffer operations";
    } else {
      CheckSignedArithmetic(expr);
      CheckSignedArithmetic(expr->operand_);
    }
    IRVisitor::VisitUnaryExpr_(expr);
  }

  void VisitExpr_(const ir::CallPtr& call) override {
    CHECK_SPAN(ir::IsOp(call, "buffer.alloc") || ir::IsOp(call, "buffer.copy") ||
                   ir::IsOp(call, "buffer.mul") || ir::IsOp(call, "buffer.set_validshape"),
               call->span_)
        << "Operation '" << call->op_->name_ << "' is not supported by direct Buffer IR codegen";
    if (ir::IsOp(call, "buffer.set_validshape")) {
      const auto type = As<ir::BufferType>(call->args_[0]->GetType());
      // PTOAS mutates both native valid fields. A fixed field, including the
      // implicit row of rank-1 buffers, has no mutable runtime counterpart.
      CHECK_SPAN(type->valid_shape_.size() == 2 && type->valid_shape_[0] == -1 && type->valid_shape_[1] == -1,
                 call->span_)
          << "Direct Buffer IR set_validshape requires a rank-2 buffer with both valid dimensions dynamic";
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  static void CheckSignedArithmetic(const ir::ExprPtr& expr) {
    CHECK_SPAN(!ir::GetScalarDtype(expr).IsUnsignedInt(), expr->span_)
        << "Unsigned scalar arithmetic is not supported by direct Buffer IR codegen; "
           "use INDEX or signed integer arithmetic";
  }

  bool returned_ = false;
  size_t depth_ = 0;
  std::unordered_set<const ir::BufferType*> descriptors_;
};

}  // namespace

bool PTOCodegen::UsesBufferIR(const ir::FunctionPtr& func) {
  BufferFunctionDetector detector;
  detector.VisitFunction(func);
  return detector.found;
}

void PTOCodegen::GenerateBufferFunction(const ir::FunctionPtr& func) {
  auto program =
      std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{func}, "BufferCodegen", func->span_);
  std::vector<Diagnostic> diagnostics;
  ir::CreateBufferIRPropertyVerifier()->Verify(program, diagnostics);
  for (const auto& diagnostic : diagnostics) {
    CHECK_SPAN(diagnostic.severity != DiagnosticSeverity::Error, diagnostic.span)
        << "Invalid Buffer IR (" << diagnostic.rule_name << "): " << diagnostic.message;
  }
  CHECK_SPAN(func->return_types_.empty(), func->span_)
      << "Direct Buffer IR codegen does not yet support function results";
  for (const auto& param : func->params_) {
    CHECK_SPAN(As<ir::ScalarType>(param->GetType()), param->span_)
        << "Direct Buffer IR codegen currently supports only scalar parameters; "
           "the device buffer-parameter ABI is not yet implemented";
  }
  BufferEmissionPreflight preflight;
  preflight.VisitFunction(func);

  fs_.Reset();
  fs_.buffer_ir = true;
  fs_.current_function = func;
  stream_ << "  func.func @" << func->name_ << "(";
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const std::string name = "%arg" + std::to_string(i);
    fs_.used_ssa_names.insert(name.substr(1));
    BindVarToMlir(func->params_[i], name);
    if (i != 0) stream_ << ", ";
    stream_ << name << ": " << GetTypeString(As<ir::ScalarType>(func->params_[i]->GetType())->dtype_);
  }
  stream_ << ")";
  if (func->func_type_ == ir::FunctionType::AIC) {
    stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<cube>}";
  } else if (func->func_type_ == ir::FunctionType::AIV) {
    stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<vector>}";
  }
  stream_ << " {\n";
  ++indent_level_;
  fs_.constants_indent = GetIndent();
  auto saved_stream = std::move(stream_);
  stream_ = std::move(fs_.body_section);
  if (func->body_) VisitStmt(func->body_);
  const std::string body = stream_.str();
  stream_ = std::move(saved_stream);
  stream_ << fs_.constants_section.str() << body << GetIndent() << "return\n";
  --indent_level_;
  stream_ << "  }\n";
}

std::string PTOCodegen::EmitBufferIntegerOperand(const ir::ExprPtr& expr, DataType target) {
  if (auto constant = As<ir::ConstInt>(expr)) return GetOrEmitConstant(constant->value_, target);
  const auto type = As<ir::ScalarType>(expr->GetType());
  INTERNAL_CHECK_SPAN(type, expr->span_) << "Internal error: buffer integer operand must be scalar";
  std::string value = GetExprAsCode(expr);
  if (type->dtype_ == target) return value;
  std::string from = GetTypeString(type->dtype_);
  if (type->dtype_.IsUnsignedInt()) {
    const std::string signless = NewTemp();
    const std::string signless_type = from.substr(1);
    Emit(signless + " = builtin.unrealized_conversion_cast " + value + " : " + from + " to " + signless_type);
    value = signless;
    from = signless_type;
    if (target == DataType::INT64 && type->dtype_.GetBit() == 64) return value;
    if (target == DataType::INDEX && type->dtype_.GetBit() < 64) {
      // index_cast sign-extends its input. Preserve the high bit of a narrow
      // unsigned extent (for example UINT8 valid_col=200) before that cast.
      const std::string widened = NewTemp();
      Emit(widened + " = arith.extui " + value + " : " + from + " to i64");
      value = widened;
      from = "i64";
    }
  }
  const std::string result = NewTemp();
  const bool index_cast = target == DataType::INDEX || type->dtype_ == DataType::INDEX;
  const std::string cast =
      index_cast ? "arith.index_cast" : (type->dtype_.IsUnsignedInt() ? "arith.extui" : "arith.extsi");
  Emit(result + " = " + cast + " " + value + " : " + from + " to " + GetTypeString(target));
  return result;
}

bool PTOCodegen::TryEmitBufferCall(const ir::CallPtr& call, const ir::VarPtr& result) {
  if (!call) return false;
  const auto& registry = ir::OpRegistry::GetInstance();
  if (As<ir::GlobalVar>(call->op_) || !registry.IsRegistered(call->op_->name_) ||
      registry.GetEntry(call->op_->name_).GetIRStage() != ir::OpIRStage::Buffer) {
    return false;
  }
  SpanScope call_loc(this, &call->span_);
  if (ir::IsOp(call, "buffer.alloc")) {
    INTERNAL_CHECK_SPAN(result, call->span_) << "Internal error: buffer.alloc needs its explicit SSA result";
    const auto type = As<ir::BufferType>(result->GetType());
    INTERNAL_CHECK_SPAN(type, call->span_) << "Internal error: buffer.alloc result must be BufferType";
    const auto valid = As<ir::MakeTuple>(call->args_[0]);
    // Native allocation operands are present only for dynamic type dimensions.
    // Static extents (including the implicit row of a rank-1 buffer) live in
    // the immutable descriptor; supplying both representations is invalid PTO.
    std::vector<std::string> dimensions(2);
    size_t dynamic = 0;
    const size_t first_native_axis = 2 - type->shape_.size();
    for (size_t axis = 0; axis < type->valid_shape_.size(); ++axis) {
      if (type->valid_shape_[axis] == -1) {
        dimensions[first_native_axis + axis] =
            EmitBufferIntegerOperand(valid->elements_[dynamic++], DataType::INDEX);
      }
    }
    const std::string address =
        call->args_.size() == 2 ? EmitBufferIntegerOperand(call->args_[1], DataType::INT64) : "";
    const std::string name = NewNamedTemp(result->name_hint_);
    const std::string descriptor = BufferTypeString(type, call->span_);
    Emit(name + " = pto.alloc_tile" + (address.empty() ? "" : " addr = " + address) +
         (dimensions[0].empty() ? "" : " valid_row = " + dimensions[0]) +
         (dimensions[1].empty() ? "" : " valid_col = " + dimensions[1]) + " : " + descriptor);
    BindVarToMlir(result, name);
    RegisterTileBufType(name, descriptor);
  } else if (ir::IsOp(call, "buffer.set_validshape")) {
    const auto shape = As<ir::MakeTuple>(call->args_[1]);
    std::vector<std::string> dimensions;
    if (shape->elements_.size() == 1) dimensions.push_back(GetOrEmitConstant(int64_t{1}, DataType::INDEX));
    for (const auto& extent : shape->elements_) {
      dimensions.push_back(EmitBufferIntegerOperand(extent, DataType::INDEX));
    }
    Emit("pto.set_validshape " + GetExprAsCode(call->args_[0]) + ", " + dimensions[0] + ", " + dimensions[1] +
         " : " + GetExprTypeAnnotation(call->args_[0]));
  } else {
    INTERNAL_CHECK_SPAN(ir::IsOp(call, "buffer.copy") || ir::IsOp(call, "buffer.mul"), call->span_)
        << "Internal error: missing preflighted buffer emitter for " << call->op_->name_;
    const size_t input_count = call->args_.size() - 1;
    std::ostringstream line;
    line << (ir::IsOp(call, "buffer.copy") ? "pto.tmov" : "pto.tmul") << " ins(";
    for (size_t i = 0; i < input_count; ++i) {
      if (i != 0) line << ", ";
      line << GetExprAsCode(call->args_[i]);
    }
    line << " : ";
    for (size_t i = 0; i < input_count; ++i) {
      if (i != 0) line << ", ";
      line << GetExprTypeAnnotation(call->args_[i]);
    }
    line << ") outs(" << GetExprAsCode(call->args_.back()) << " : "
         << GetExprTypeAnnotation(call->args_.back()) << ")";
    Emit(line.str());
  }
  fs_.current_expr_value.clear();
  return true;
}

}  // namespace codegen
}  // namespace pypto
