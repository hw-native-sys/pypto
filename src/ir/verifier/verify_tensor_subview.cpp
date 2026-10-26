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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tensor_subview.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"

namespace pypto::ir {
namespace {
class BorrowedTypeCache {
 public:
  bool Contains(const TypePtr& type) {
    auto found = cache_.find(type.get());
    if (found != cache_.end()) return found->second;
    bool borrowed = IsAffineSubviewType(type);
    if (auto tuple = As<TupleType>(type)) {
      for (const auto& field : tuple->types_) borrowed |= Contains(field);
    }
    cache_[type.get()] = borrowed;
    return borrowed;
  }

 private:
  std::unordered_map<const Type*, bool> cache_;
};

// One ordered walk, indexed provenance, scoped arithmetic facts: O(N) visits.
// Roots are caller-owned parameters. Borrowed descriptors cannot become roots
// by merely acquiring an AffineSubview type (including through deserialization).
class SubviewVerifier : public IRVisitor {
 public:
  SubviewVerifier(const FunctionPtr& function, const backend::BackendHandler* handler)
      : function_(function), handler_(handler), analyzer_(std::make_shared<arith::Analyzer>()) {
    for (const auto& param : function->params_) {
      CHECK_SPAN(!ContainsBorrowedTensorType(param->GetType()), param->span_)
          << "tensor.subview descriptors cannot be function parameters";
      if (As<TensorType>(param->GetType())) roots_.insert(param.get());
    }
    for (const auto& result : function->return_types_) {
      CHECK_SPAN(!ContainsBorrowedTensorType(result), function->span_)
          << "tensor.subview cannot escape as a function result";
    }
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    VisitExpr(op->value_);
    auto call = As<Call>(op->value_);
    bool alias = roots_.count(op->value_.get()) != 0;
    if (call &&
        (IsOp(call, "tensor.subview") || IsOp(call, "tensor.view") || IsOp(call, "tensor.set_validshape"))) {
      alias = !call->args_.empty() && roots_.count(call->args_[0].get());
    }
    CHECK_SPAN(!IsAffineSubviewType(op->value_->GetType()) || IsAffineSubviewType(op->var_->GetType()),
               op->span_)
        << "tensor.subview assignment cannot discard affine addressing";
    if (IsAffineSubviewType(op->var_->GetType())) {
      auto tensor = As<TensorType>(op->var_->GetType());
      CHECK_SPAN(tensor, op->span_) << "tensor.subview requires an ordinary Tensor alias";
      CHECK_SPAN(!tensor->memref_, op->span_) << "tensor.subview borrowed descriptor must not own a MemRef";
      CHECK_SPAN(alias, op->span_) << "tensor.subview requires verified caller-owned GM provenance";
      CHECK_SPAN(IsAffineSubviewType(op->value_->GetType()), op->span_)
          << "tensor.subview alias lost its addressing contract";
      CheckBorrowedType(op->var_->GetType(), op->value_->GetType(), op->span_);
    }
    if (alias) {
      roots_.insert(op->var_.get());
      local_roots_.push_back(op->var_.get());
    }
    if (auto scalar = As<ScalarType>(op->var_->GetType());
        scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX)) {
      scalar_arithmetic_safe_[op->var_.get()] = ScalarArithmeticIsSafe(op->value_);
      analyzer_->Bind(op->var_, op->value_);
    }
  }

  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "tensor.subview")) {
      CheckSubview(op);
    }
    for (size_t i = 0; i < op->args_.size(); ++i) {
      const auto& arg = op->args_[i];
      CHECK_SPAN(!ContainsBorrowedTensorType(arg->GetType()) || IsAffineSubviewType(arg->GetType()),
                 op->span_)
          << "tensor.subview tuple consumers are not supported";
      if (!IsAffineSubviewType(arg->GetType())) continue;
      CHECK_SPAN(roots_.count(arg.get()), op->span_)
          << "tensor.subview descriptor is outside its defining scope or lacks provenance";
      CHECK_SPAN(!As<GlobalVar>(op->op_), op->span_)
          << "tensor.subview cannot cross a non-inlined call boundary";
      auto& registry = OpRegistry::GetInstance();
      const auto& entry = registry.GetEntry(op->op_->name_);
      CHECK_SPAN(entry.GetIRStage() == OpIRStage::Functional, op->span_)
          << "tensor.subview borrowed descriptors are not supported by Buffer IR consumers";
      CHECK_SPAN(!ArgEffectWrites(entry.GetArgEffect(i, op->kwargs_)), op->span_)
          << "tensor.subview is read-only; writer " << op->op_->name_ << " is not supported";
      // Pure metadata aliases must preserve the tag. Other tensor operations
      // materialize results normally and must not inherit borrowed ownership.
      if (IsOp(op, "tensor.view") || IsOp(op, "tensor.set_validshape")) {
        CHECK_SPAN(IsAffineSubviewType(op->GetType()), op->span_)
            << "tensor.subview metadata alias must preserve affine addressing";
        auto deduced = registry.Create(op->op_->name_, op->args_, op->kwargs_, op->span_);
        CheckBorrowedType(op->GetType(), deduced->GetType(), op->span_);
        if (IsOp(op, "tensor.set_validshape")) CheckValidShape(op);
        if (IsOp(op, "tensor.view")) CheckView(op);
      }
      CHECK_SPAN(entry.GetOpCategory() == "TensorOp" || IsOp(op, "tile.load"), op->span_)
          << "tensor.subview supports tensor read consumers and explicit Vec loads only";
      CHECK_SPAN(!IsOp(op, "tensor.reshape") && !IsOp(op, "tensor.transpose") &&
                     !IsOp(op, "tensor.reinterpret_view"),
                 op->span_)
          << "tensor.subview requires an explicit load before " << op->op_->name_;
      CHECK_SPAN(!handler_ || IsOp(op, "tensor.subview") || IsOp(op, "tensor.view") ||
                     IsOp(op, "tensor.set_validshape") || IsOp(op, "tensor.dim") || IsOp(op, "tile.load"),
                 op->span_)
          << "tensor.subview backend supports metadata and Vec-load consumers only";
      if (IsOp(op, "tile.load")) CheckLoad(op);
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    for (const auto& arg : op->args_) {
      CHECK_SPAN(!ContainsBorrowedTensorType(arg->GetType()), op->span_)
          << "tensor.subview cannot cross a submit boundary";
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ReturnStmtPtr& op) override { CheckEscape(op->value_, op->span_); }
  void VisitStmt_(const YieldStmtPtr& op) override {
    CheckEscape(op->value_, op->span_);
    yielded_bounds_.clear();
    yielded_safe_.clear();
    for (const auto& value : op->value_) {
      yielded_bounds_.push_back(analyzer_->const_int_bound(value));
      yielded_safe_.push_back(ScalarArithmeticIsSafe(value));
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    for (const auto& arg : op->iter_args_) {
      CHECK_SPAN(!ContainsBorrowedTensorType(arg->GetType()) &&
                     !ContainsBorrowedTensorType(arg->initValue_->GetType()),
                 op->span_)
          << "tensor.subview cannot be loop carried";
    }
    auto start = analyzer_->const_int_bound(op->start_);
    auto stop = analyzer_->const_int_bound(op->stop_);
    auto step = analyzer_->const_int_bound(op->step_);
    const bool safe_bounds = ScalarArithmeticIsSafe(op->start_) && ScalarArithmeticIsSafe(op->stop_) &&
                             ScalarArithmeticIsSafe(op->step_);
    scalar_arithmetic_safe_[op->loop_var_.get()] = safe_bounds;
    // Analyzer ranges are mathematical. Never bind an induction variable from
    // a narrow expression that may already have wrapped before index widening.
    if (safe_bounds && start.min_value > arith::ConstIntBound::kNegInf &&
        stop.max_value < arith::ConstIntBound::kPosInf && step.min_value > 0 &&
        start.min_value < stop.max_value) {
      analyzer_->Bind(op->loop_var_, start.min_value, stop.max_value);
    }
    auto saved = local_roots_.size();
    VisitStmt(op->body_);
    LeaveScope(saved);
    analyzer_->Unbind(op->loop_var_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& arg : op->iter_args_) {
      CHECK_SPAN(!ContainsBorrowedTensorType(arg->GetType()) &&
                     !ContainsBorrowedTensorType(arg->initValue_->GetType()),
                 op->span_)
          << "tensor.subview cannot be loop carried";
    }
    auto saved = local_roots_.size();
    {
      auto scope = SafeConstraintContext(op->condition_);
      VisitStmt(op->body_);
    }
    LeaveScope(saved);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    auto saved = local_roots_.size();
    auto outer_yields = std::move(yielded_bounds_);
    auto outer_safe = std::move(yielded_safe_);
    yielded_bounds_.clear();
    yielded_safe_.clear();
    {
      auto scope = SafeConstraintContext(op->condition_);
      VisitStmt(op->then_body_);
    }
    auto then_bounds = std::move(yielded_bounds_);
    auto then_safe = std::move(yielded_safe_);
    yielded_bounds_.clear();
    yielded_safe_.clear();
    LeaveScope(saved);
    if (op->else_body_) {
      auto scope = SafeConstraintContext(MakeNot(op->condition_));
      VisitStmt(*op->else_body_);
      LeaveScope(saved);
    }
    // Scalar phis carry the union of both branch ranges. Descriptor phis are
    // rejected by CheckEscape; geometry can be merged before constructing one.
    if (then_bounds.size() == op->return_vars_.size() && yielded_bounds_.size() == op->return_vars_.size()) {
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        auto var = op->return_vars_[i];
        if (!As<ScalarType>(var->GetType())) continue;
        scalar_arithmetic_safe_[var.get()] = then_safe[i] && yielded_safe_[i];
        const auto lo = std::min(then_bounds[i].min_value, yielded_bounds_[i].min_value);
        const auto hi = std::max(then_bounds[i].max_value, yielded_bounds_[i].max_value);
        if (lo > arith::ConstIntBound::kNegInf && hi < arith::ConstIntBound::kPosInf - 1) {
          analyzer_->Bind(var, lo, hi + 1);
        }
      }
    }
    yielded_bounds_ = std::move(outer_yields);
    yielded_safe_ = std::move(outer_safe);
  }

 private:
  bool ContainsBorrowedTensorType(const TypePtr& type) { return borrowed_types_.Contains(type); }
  std::optional<arith::ConstraintContext> SafeConstraintContext(const ExprPtr& condition) {
    // A predicate is evidence only when its original machine arithmetic agrees
    // with the mathematical expressions modeled by Analyzer. Untrusted guards
    // contribute no facts; the usual prove-or-reject checks remain in force.
    if (!ScalarArithmeticIsSafe(condition)) return std::nullopt;
    return analyzer_->GetConstraintContext(condition);
  }
  // The range analyzer models Add/Sub/Mul mathematically. A widening cast
  // cannot repair overflow that already happened in a narrower operand.
  // Record scalar definitions once; do not rescan alias chains at every use.
  bool ScalarArithmeticIsSafe(const ExprPtr& expression) {
    class CheckArithmetic : public IRVisitor {
     public:
      CheckArithmetic(arith::Analyzer& analyzer, const std::unordered_map<const Expr*, bool>& definitions)
          : analyzer_(analyzer), definitions_(definitions) {}
      bool safe = true;
      void VisitExpr(const ExprPtr& expr) override {
        if (!safe || !seen_.insert(expr.get()).second) return;
        if (auto var = AsVarLike(expr)) {
          auto it = definitions_.find(var.get());
          if (it != definitions_.end()) safe &= it->second;
          return;
        }
        auto scalar = As<ScalarType>(expr->GetType());
        bool widening_cast = false;
        if (auto cast = As<Cast>(expr)) {
          auto operand = As<ScalarType>(cast->operand_->GetType());
          if (operand && scalar && operand->dtype_.IsInt() && scalar->dtype_.IsInt()) {
            const auto from_bits = operand->dtype_.GetBit();
            const auto to_bits = scalar->dtype_.GetBit();
            widening_cast =
                (scalar->dtype_.IsSignedInt() == operand->dtype_.IsSignedInt() && to_bits >= from_bits) ||
                (operand->dtype_.IsUnsignedInt() && scalar->dtype_.IsSignedInt() && to_bits > from_bits);
          }
        }
        if (scalar && scalar->dtype_.IsInt() && !As<Call>(expr) && !widening_cast) {
          auto bound = analyzer_.const_int_bound(expr);
          const auto bits = scalar->dtype_.GetBit();
          int64_t lo = arith::ConstIntBound::kNegInf + 1;
          int64_t hi = arith::ConstIntBound::kPosInf - 1;
          if (scalar->dtype_.IsUnsignedInt()) lo = 0;
          if (bits > 0 && bits < 64) {
            if (scalar->dtype_.IsUnsignedInt()) {
              hi = (int64_t{1} << bits) - 1;
            } else {
              lo = -(int64_t{1} << (bits - 1));
              hi = (int64_t{1} << (bits - 1)) - 1;
            }
          }
          safe &= bound.min_value >= lo && bound.max_value <= hi;
        }
        IRVisitor::VisitExpr(expr);
      }

     private:
      arith::Analyzer& analyzer_;
      const std::unordered_map<const Expr*, bool>& definitions_;
      std::unordered_set<const Expr*> seen_;
    } check(*analyzer_, scalar_arithmetic_safe_);
    check.VisitExpr(expression);
    return check.safe;
  }
  void LeaveScope(size_t begin) {
    for (size_t i = begin; i < local_roots_.size(); ++i) roots_.erase(local_roots_[i]);
    local_roots_.resize(begin);
  }
  void CheckBorrowedType(const TypePtr& actual, const TypePtr& expected, const Span& span) {
    auto a = As<TensorType>(actual);
    auto e = As<TensorType>(expected);
    CHECK_SPAN(a && e && IsAffineSubviewType(actual) && IsAffineSubviewType(expected) && !a->memref_ &&
                   a->dtype_ == e->dtype_ && a->shape_.size() == e->shape_.size() &&
                   a->tensor_view_->layout == e->tensor_view_->layout &&
                   a->tensor_view_->pad == e->tensor_view_->pad &&
                   a->tensor_view_->stride.size() == e->tensor_view_->stride.size() &&
                   a->tensor_view_->valid_shape.size() == e->tensor_view_->valid_shape.size(),
               span)
        << "tensor.subview alias type does not preserve its defining address map";
    auto compare = [&](const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
      for (size_t i = 0; i < lhs.size(); ++i) {
        Prove(MakeEq(lhs[i], rhs[i]), span, "alias geometry equality");
      }
    };
    compare(a->shape_, e->shape_);
    compare(a->tensor_view_->stride, e->tensor_view_->stride);
    compare(a->tensor_view_->valid_shape, e->tensor_view_->valid_shape);
  }
  void CheckEscape(const std::vector<ExprPtr>& values, const Span& span) {
    for (const auto& value : values) {
      CHECK_SPAN(!ContainsBorrowedTensorType(value->GetType()), span)
          << "tensor.subview cannot escape through return/yield or a geometry phi";
      VisitExpr(value);
    }
  }
  void Prove(const ExprPtr& condition, const Span& span, const char* obligation) {
    CHECK_SPAN(analyzer_->CanProve(condition), span)
        << "tensor.subview cannot prove " << obligation << "; false and unknown bounds are rejected";
  }
  void CheckFinite(const ExprPtr& value, const Span& span) {
    const auto bound = analyzer_->const_int_bound(value);
    CHECK_SPAN(bound.min_value >= 0 && bound.max_value < arith::ConstIntBound::kPosInf, span)
        << "tensor.subview cannot prove nonnegative overflow-free index arithmetic";
    if (handler_) {
      CHECK_SPAN(bound.max_value <= handler_->GetSubviewLoadLimits().max_index, span)
          << "tensor.subview exceeds the supported device index/DMA range";
    }
  }
  void CheckSubview(const CallPtr& op) {
    CHECK_SPAN(function_->ir_stage_ == FunctionIRStage::Functional, op->span_)
        << "tensor.subview requires the functional Tensor/Tile compilation profile";
    CHECK_SPAN(IsInCoreType(function_->func_type_), op->span_)
        << "tensor.subview is supported only inside InCore functions";
    CHECK_SPAN(op->args_.size() == 4 && roots_.count(op->args_[0].get()), op->span_)
        << "tensor.subview requires a caller-owned ordinary GM Tensor root";
    auto parent = As<TensorType>(op->args_[0]->GetType());
    auto result = As<TensorType>(op->GetType());
    CHECK_SPAN(parent && result && IsAffineSubviewType(op->GetType()), op->span_)
        << "tensor.subview requires ordinary Tensor types with affine result addressing";
    auto shape = As<MakeTuple>(op->args_[1]);
    auto offsets = As<MakeTuple>(op->args_[2]);
    auto steps = As<MakeTuple>(op->args_[3]);
    const auto rank = parent->shape_.size();
    CHECK_SPAN(shape && offsets && steps && shape->elements_.size() == rank &&
                   offsets->elements_.size() == rank && steps->elements_.size() == rank && rank > 0,
               op->span_)
        << "tensor.subview geometry rank mismatch";
    CHECK_SPAN(!handler_ || handler_->GetSubviewLoadLimits().max_rows > 0, op->span_)
        << "tensor.subview native loads are unsupported on this backend";
    auto parent_view = parent->tensor_view_.value_or(TensorView{});
    CHECK_SPAN(parent_view.pad == PadValue::null &&
                   (parent_view.layout == TensorLayout::ND || parent_view.layout == TensorLayout::DN),
               op->span_)
        << "tensor.subview requires unpadded ND/DN parent addressing";
    if (!IsAffineSubviewType(parent)) {
      auto canonical = tensor_view_semantics::CheckCanonicalView(
          parent->shape_,
          parent_view.stride.empty()
              ? tensor_view_semantics::BuildLogicalStridesFromLayout(parent->shape_, parent_view.layout)
              : parent_view.stride,
          parent_view.layout, true);
      CHECK_SPAN(canonical.ok, op->span_)
          << "tensor.subview requires a canonical GM root: " << canonical.reason;
    }
    CHECK_SPAN(result->dtype_ == parent->dtype_ && result->tensor_view_->layout == parent_view.layout &&
                   result->tensor_view_->pad == PadValue::null && result->tensor_view_->valid_shape.empty() &&
                   !result->memref_,
               op->span_)
        << "tensor.subview result must preserve parent dtype/layout with a fully valid borrowed window";
    const auto& extent = parent_view.valid_shape.empty() ? parent->shape_ : parent_view.valid_shape;
    auto strides = parent_view.stride.empty() ? tensor_view_semantics::BuildLogicalStridesFromLayout(
                                                    parent->shape_, parent_view.layout)
                                              : parent_view.stride;
    CHECK_SPAN(extent.size() == rank && strides.size() == rank, op->span_)
        << "tensor.subview parent metadata rank mismatch";
    std::vector<ExprPtr> geometry = shape->elements_;
    geometry.insert(geometry.end(), offsets->elements_.begin(), offsets->elements_.end());
    geometry.insert(geometry.end(), steps->elements_.begin(), steps->elements_.end());
    geometry.insert(geometry.end(), strides.begin(), strides.end());
    geometry.insert(geometry.end(), extent.begin(), extent.end());
    CHECK_SPAN(ScalarArithmeticIsSafe(std::make_shared<MakeTuple>(geometry, op->span_)), op->span_)
        << "tensor.subview cannot prove overflow-free geometry arithmetic in its original dtype";
    auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, op->span_);
    auto one = std::make_shared<ConstInt>(1, DataType::INDEX, op->span_);
    bool strict = false;
    ExprPtr max_address = zero;
    for (size_t i = 0; i < rank; ++i) {
      const auto& size = shape->elements_[i];
      const auto& offset = offsets->elements_[i];
      const auto& step = steps->elements_[i];
      Prove(MakeGt(size, zero), op->span_, "shape > 0");
      Prove(MakeGt(step, zero), op->span_, "steps > 0");
      Prove(MakeGe(offset, zero), op->span_, "offsets >= 0");
      CheckFinite(size, op->span_);
      CheckFinite(step, op->span_);
      CheckFinite(offset, op->span_);
      auto span = MakeMul(MakeSub(size, one), step);
      CheckFinite(span, op->span_);
      auto last = MakeAdd(offset, span);
      CheckFinite(last, op->span_);
      Prove(MakeLt(last, extent[i]), op->span_, "sampled window inside immediate parent");
      strict |= analyzer_->CanProve(MakeLt(size, extent[i]));
      CheckFinite(strides[i], op->span_);
      auto pitch = MakeMul(strides[i], step);
      CheckFinite(pitch, op->span_);
      CHECK_SPAN(result->shape_.size() == rank && result->tensor_view_->stride.size() == rank &&
                     analyzer_->CanProveEqual(result->shape_[i], size) &&
                     analyzer_->CanProveEqual(result->tensor_view_->stride[i], pitch),
                 op->span_)
          << "tensor.subview result metadata does not match its defining geometry";
      auto address = MakeMul(last, strides[i]);
      CheckFinite(address, op->span_);
      max_address = MakeAdd(max_address, address);
      CheckFinite(max_address, op->span_);
    }
    CHECK_SPAN(strict, op->span_) << "tensor.subview cannot prove strictly smaller parent window";
    CheckFinite(
        MakeMul(max_address, std::make_shared<ConstInt>(static_cast<int64_t>(parent->dtype_.GetByte()),
                                                        DataType::INDEX, op->span_)),
        op->span_);
    if (handler_) {
      CHECK_SPAN(rank == 2 && parent_view.layout == TensorLayout::ND && parent->dtype_ == DataType::BF16 &&
                     analyzer_->CanProveEqual(result->tensor_view_->stride[1], one),
                 op->span_)
          << "tensor.subview backend profile requires rank-2 ND BF16 with unit inner stride";
    }
  }
  void CheckView(const CallPtr& op) {
    auto parent = As<TensorType>(op->args_[0]->GetType());
    auto result = As<TensorType>(op->GetType());
    // Ordinary view deduction deliberately defers symbolic product equality.
    // Borrowed windows cannot inherit that optimism: a symbolic reshape must
    // prove the same storage footprint before its new extents can bound loads.
    auto count = [&](const std::vector<ExprPtr>& shape) {
      ExprPtr product = std::make_shared<ConstInt>(1, DataType::INDEX, op->span_);
      for (const auto& dim : shape) {
        CHECK_SPAN(ScalarArithmeticIsSafe(dim), op->span_)
            << "tensor.subview cannot prove overflow-free view shape arithmetic";
        Prove(MakeGt(dim, std::make_shared<ConstInt>(0, DataType::INDEX, op->span_)), op->span_,
              "view shape > 0");
        product = MakeMul(product, dim);
        CheckFinite(product, op->span_);
      }
      return product;
    };
    Prove(MakeEq(count(parent->shape_), count(result->shape_)), op->span_,
          "view preserves parent element count");
  }
  void CheckValidShape(const CallPtr& op) {
    auto parent = As<TensorType>(op->args_[0]->GetType());
    auto result = As<TensorType>(op->GetType());
    const auto& extent =
        parent->tensor_view_->valid_shape.empty() ? parent->shape_ : parent->tensor_view_->valid_shape;
    CHECK_SPAN(result->shape_.size() == parent->shape_.size() &&
                   result->tensor_view_->stride.size() == parent->tensor_view_->stride.size() &&
                   result->tensor_view_->layout == parent->tensor_view_->layout &&
                   result->tensor_view_->pad == parent->tensor_view_->pad,
               op->span_)
        << "tensor.subview set_validshape must preserve the address map";
    for (size_t i = 0; i < parent->shape_.size(); ++i) {
      Prove(MakeEq(result->shape_[i], parent->shape_[i]), op->span_, "metadata alias shape equality");
      Prove(MakeEq(result->tensor_view_->stride[i], parent->tensor_view_->stride[i]), op->span_,
            "metadata alias stride equality");
    }
    const auto& result_valid =
        result->tensor_view_->valid_shape.empty() ? result->shape_ : result->tensor_view_->valid_shape;
    CHECK_SPAN(result_valid.size() == extent.size(), op->span_)
        << "tensor.subview set_validshape rank mismatch";
    for (size_t i = 0; i < extent.size(); ++i) {
      auto dim = result_valid[i];
      Prove(MakeGt(dim, std::make_shared<ConstInt>(0, DataType::INDEX, op->span_)), op->span_,
            "valid shape > 0");
      Prove(MakeLe(dim, extent[i]), op->span_, "valid shape narrows the parent");
    }
  }
  void CheckLoad(const CallPtr& op) {
    if (!handler_) return;
    auto source = As<TensorType>(op->args_[0]->GetType());
    auto one = std::make_shared<ConstInt>(1, DataType::INDEX, op->span_);
    CHECK_SPAN(source && source->shape_.size() == 2 && source->dtype_ == DataType::BF16 &&
                   source->tensor_view_->layout == TensorLayout::ND &&
                   analyzer_->CanProveEqual(source->tensor_view_->stride[1], one),
               op->span_)
        << "tensor.subview backend load requires rank-2 ND BF16 with unit inner stride";
    auto tile = As<TileType>(op->GetType());
    CHECK_SPAN(tile && tile->shape_.size() == 2 && tile->memory_space_ == MemorySpace::Vec, op->span_)
        << "tensor.subview backend profile supports rank-2 Vec loads only";
    const auto limits = handler_->GetSubviewLoadLimits();
    auto ci = [&](int64_t value) { return std::make_shared<ConstInt>(value, DataType::INDEX, op->span_); };
    const auto& valid = tile->tile_view_ && !tile->tile_view_->valid_shape.empty()
                            ? tile->tile_view_->valid_shape
                            : tile->shape_;
    Prove(MakeLe(tile->shape_[0], ci(limits.max_rows)), op->span_, "native row-count limit");
    Prove(MakeGt(valid[0], ci(0)), op->span_, "native valid rows > 0");
    Prove(MakeGt(valid[1], ci(0)), op->span_, "native valid columns > 0");
    auto offsets = As<MakeTuple>(op->args_[1]);
    CHECK_SPAN(offsets && offsets->elements_.size() == 2, op->span_)
        << "tensor.subview native load offsets must have rank 2";
    const auto& extent =
        source->tensor_view_->valid_shape.empty() ? source->shape_ : source->tensor_view_->valid_shape;
    ExprPtr last_address = ci(0);
    for (size_t i = 0; i < 2; ++i) {
      auto offset = offsets->elements_[i];
      CHECK_SPAN(ScalarArithmeticIsSafe(offset) && ScalarArithmeticIsSafe(valid[i]) &&
                     ScalarArithmeticIsSafe(tile->shape_[i]),
                 op->span_)
          << "tensor.subview cannot prove overflow-free native load geometry";
      Prove(MakeLe(valid[i], tile->shape_[i]), op->span_, "native validity inside the Tile allocation");
      Prove(MakeGe(offset, ci(0)), op->span_, "native load offsets >= 0");
      Prove(MakeLe(MakeAdd(offset, valid[i]), extent[i]), op->span_,
            "native load inside the borrowed window");
      last_address = MakeAdd(
          last_address, MakeMul(MakeSub(MakeAdd(offset, valid[i]), ci(1)), source->tensor_view_->stride[i]));
      CheckFinite(last_address, op->span_);
    }
    auto byte_count = MakeMul(valid[1], ci(2));
    auto physical_bytes = MakeMul(tile->shape_[1], ci(2));
    Prove(MakeEq(MakeFloorMod(byte_count, ci(limits.row_alignment_bytes)), ci(0)), op->span_,
          "native valid row-byte alignment");
    Prove(MakeEq(MakeFloorMod(physical_bytes, ci(limits.row_alignment_bytes)), ci(0)), op->span_,
          "native tile row-byte alignment");
    auto gap = MakeMul(MakeSub(source->tensor_view_->stride[0], valid[1]), ci(2));
    Prove(MakeGe(gap, ci(0)), op->span_, "native nonnegative source gap");
    Prove(MakeLe(gap, ci(limits.max_gap_bytes)), op->span_, "native source-gap width");
  }
  FunctionPtr function_;
  const backend::BackendHandler* handler_;
  std::shared_ptr<arith::Analyzer> analyzer_;
  std::unordered_set<const Expr*> roots_;
  std::vector<const Expr*> local_roots_;
  std::vector<arith::ConstIntBound> yielded_bounds_;
  std::vector<bool> yielded_safe_;
  std::unordered_map<const Expr*, bool> scalar_arithmetic_safe_;
  BorrowedTypeCache borrowed_types_;
};
}  // namespace
void VerifyTensorSubviews(const FunctionPtr& function, const backend::BackendHandler* backend_handler) {
  // Keep ordinary kernels off the arithmetic-analysis path. Detection is a
  // separate linear walk so false/unknown proofs are never silently skipped
  // just because the descriptor does not have a well-formed defining call.
  class Presence : public IRVisitor {
   public:
    bool found = false;
    BorrowedTypeCache types;
    void VisitVarLike_(const VarPtr& var) override { found |= types.Contains(var->GetType()); }
    void VisitExpr_(const CallPtr& call) override {
      found |= IsOp(call, "tensor.subview") || IsAffineSubviewType(call->GetType());
      IRVisitor::VisitExpr_(call);
    }
  } presence;
  for (const auto& param : function->params_) presence.found |= presence.types.Contains(param->GetType());
  for (const auto& result : function->return_types_) presence.found |= presence.types.Contains(result);
  presence.VisitStmt(function->body_);
  if (!presence.found) return;
  SubviewVerifier verifier(function, backend_handler);
  verifier.VisitStmt(function->body_);
}
}  // namespace pypto::ir
