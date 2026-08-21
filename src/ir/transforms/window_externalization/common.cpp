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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/window_externalization.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/window_externalization/internal.h"

namespace pypto {
namespace ir {
namespace window_externalization {
using transform_utils::FlattenToStmts;

namespace {

class VarRefCounter : public IRVisitor {
 public:
  explicit VarRefCounter(const Var* target) : target_(target) {}

  [[nodiscard]] size_t count() const { return count_; }

 protected:
  void VisitExpr_(const VarPtr& op) override {
    if (op.get() == target_) ++count_;
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IterArgPtr& op) override {
    if (op.get() == target_) ++count_;
    IRVisitor::VisitExpr_(op);
  }

 private:
  const Var* target_;
  size_t count_ = 0;
};

/// Count Var/IterArg references to `target` inside a statement subtree.

class GeneratedScalarLocalFlattener : public IRMutator {
 public:
  GeneratedScalarLocalFlattener(std::string name_prefix, WindowRewriteContext& rewrite_context,
                                std::vector<StmtPtr>* stmts, Span span)
      : name_prefix_(std::move(name_prefix)),
        rewrite_context_(rewrite_context),
        stmts_(stmts),
        span_(std::move(span)) {}

  ExprPtr Flatten(const ExprPtr& expr) { return VisitExpr(expr); }

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override { return ExtractCallToTemp(IRMutator::VisitExpr_(op)); }

  ExprPtr VisitExpr_(const SubmitPtr& op) override { return ExtractCallToTemp(IRMutator::VisitExpr_(op)); }

 private:
  ExprPtr ExtractCallToTemp(const ExprPtr& expr) {
    if (!As<Call>(expr) && !As<Submit>(expr)) return expr;
    auto temp_var = std::make_shared<Var>(rewrite_context_.NextScalarTempName(name_prefix_), expr->GetType(),
                                          expr->span_);
    stmts_->push_back(std::make_shared<AssignStmt>(temp_var, expr, temp_var->span_));
    return temp_var;
  }

  std::string name_prefix_;
  WindowRewriteContext& rewrite_context_;
  std::vector<StmtPtr>* stmts_;
  Span span_;
};

/// `ceil(distance / step_abs)` for a non-negative `distance` and a positive
/// `step_abs`, or nullopt when the round-up would overflow int64.
std::optional<int64_t> CheckedCeilDiv(int64_t distance, int64_t step_abs) {
  auto sum = CheckedAdd(distance, step_abs);
  if (!sum.has_value()) return std::nullopt;
  auto rounded = CheckedSub(*sum, 1);
  if (!rounded.has_value()) return std::nullopt;
  return *rounded / step_abs;
}

}  // namespace

std::string GetCallFuncName(const CallPtr& call) {
  auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
  return gvar ? gvar->name_ : "";
}

std::vector<OutParamReturnMapping> BuildOutParamReturnMappings(const FunctionPtr& func, bool include_inout) {
  // Collect output param vars and their indices.
  std::unordered_map<const Var*, size_t> out_var_to_param_idx;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const bool is_output = i < func->param_directions_.size() &&
                           (func->param_directions_[i] == ParamDirection::Out ||
                            (include_inout && func->param_directions_[i] == ParamDirection::InOut));
    if (is_output) {
      out_var_to_param_idx[func->params_[i].get()] = i;
    }
  }
  if (out_var_to_param_idx.empty()) return {};

  auto body_stmts = FlattenToStmts(func->body_);

  // Build var->assign map for quick lookup.
  std::unordered_map<const Var*, AssignStmtPtr> var_def;
  for (const auto& stmt : body_stmts) {
    if (auto assign = As<AssignStmt>(stmt)) {
      var_def[assign->var_.get()] = assign;
    }
  }

  std::unordered_map<const Var*, ExprPtr> loop_return_to_init;
  for (const auto& stmt : body_stmts) {
    if (auto loop = As<ForStmt>(stmt)) {
      for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
        loop_return_to_init[loop->return_vars_[i].get()] = loop->iter_args_[i]->initValue_;
      }
    } else if (auto loop = As<WhileStmt>(stmt)) {
      for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
        loop_return_to_init[loop->return_vars_[i].get()] = loop->iter_args_[i]->initValue_;
      }
    }
  }

  ReturnStmtPtr return_stmt;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      return_stmt = ret;
      break;
    }
  }
  if (!return_stmt) return {};

  std::vector<OutParamReturnMapping> result;
  for (size_t ret_i = 0; ret_i < return_stmt->value_.size(); ++ret_i) {
    auto ret_var = As<Var>(return_stmt->value_[ret_i]);
    if (!ret_var) continue;

    auto def_it = var_def.find(ret_var.get());
    if (def_it == var_def.end()) {
      auto loop_it = loop_return_to_init.find(ret_var.get());
      if (loop_it == loop_return_to_init.end()) continue;
      auto init_var = AsVarLike(loop_it->second);
      if (!init_var) continue;
      auto param_it = out_var_to_param_idx.find(init_var.get());
      if (param_it == out_var_to_param_idx.end()) continue;
      result.push_back({param_it->second, ret_i, func->params_[param_it->second]});
      continue;
    }

    auto call = As<Call>(def_it->second->value_);
    if (!call || !IsOp(call, "tile.store")) continue;
    if (call->args_.size() < 3) continue;

    auto out_tensor = As<Var>(call->args_[2]);
    if (!out_tensor) continue;
    auto param_it = out_var_to_param_idx.find(out_tensor.get());
    if (param_it == out_var_to_param_idx.end()) continue;

    result.push_back({param_it->second, ret_i, func->params_[param_it->second]});
  }

  return result;
}

bool IsWindowizeEnabled(const FunctionPtr& func) { return func && func->GetAttr<bool>("windowize", false); }

std::unordered_map<std::string, FunctionPtr> BuildFunctionLookup(const ProgramPtr& program) {
  std::unordered_map<std::string, FunctionPtr> lookup;
  if (!program) return lookup;
  lookup.reserve(program->functions_.size());
  for (const auto& [gvar, func] : program->functions_) {
    if (func) lookup.emplace(func->name_, func);
  }
  return lookup;
}

size_t CountVarRefsInStmt(const StmtPtr& stmt, const Var* target) {
  VarRefCounter counter(target);
  counter.VisitStmt(stmt);
  return counter.count();
}

size_t CountVarRefsInExpr(const ExprPtr& expr, const Var* target) {
  VarRefCounter counter(target);
  counter.VisitExpr(expr);
  return counter.count();
}

bool ExprReferencesOnlyVarsIn(const ExprPtr& expr, const std::unordered_set<const Var*>& allowed) {
  class Checker : public IRVisitor {
   public:
    explicit Checker(const std::unordered_set<const Var*>& allowed) : allowed_(allowed) {}

    [[nodiscard]] bool ok() const { return ok_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

   private:
    const std::unordered_set<const Var*>& allowed_;
    bool ok_ = true;
  };

  Checker checker(allowed);
  checker.VisitExpr(expr);
  return checker.ok();
}

ExprPtr FlattenGeneratedScalarExprWithLocalTemps(const ExprPtr& expr, const std::string& name_prefix,
                                                 const Span& span, std::vector<StmtPtr>* stmts,
                                                 WindowRewriteContext& rewrite_context) {
  if (!expr || !stmts) return expr;
  GeneratedScalarLocalFlattener flattener(name_prefix, rewrite_context, stmts, span);
  return flattener.Flatten(expr);
}

AccessRegion MakeDenseRegion(std::vector<DenseRegionPiece> pieces) { return AccessRegion{std::move(pieces)}; }

const std::vector<DenseRegionPiece>& DensePieces(const OutputRewriteInfo& info) {
  return info.region.dense_pieces;
}

const std::vector<DenseRegionPiece>& DensePieces(const InputRewriteInfo& info) {
  return info.region.dense_pieces;
}

std::optional<TensorView> MakeWindowTensorView(const std::shared_ptr<const TensorType>& tensor_type,
                                               const std::vector<ExprPtr>& parent_shape,
                                               const std::vector<ExprPtr>& window_shape) {
  if (!tensor_type) return std::nullopt;
  if (tensor_type->tensor_view_.has_value()) {
    auto new_view = tensor_type->tensor_view_;
    if (new_view->stride.empty()) {
      if (new_view->layout == TensorLayout::NZ) return std::nullopt;
      new_view->stride =
          tensor_view_semantics::BuildLogicalStridesFromLayout(tensor_type->shape_, new_view->layout);
    }
    if (!new_view->valid_shape.empty()) new_view->valid_shape = window_shape;
    return new_view;
  }

  auto parent_strides = tensor_view_semantics::BuildLogicalStridesFromLayout(parent_shape, TensorLayout::ND);
  if (parent_strides.size() != window_shape.size()) return std::nullopt;
  return TensorView(std::move(parent_strides), TensorLayout::ND);
}

TypePtr MakeWindowTensorType(const std::shared_ptr<const TensorType>& tensor_type,
                             const std::vector<ExprPtr>& parent_shape,
                             const std::vector<ExprPtr>& window_shape) {
  auto new_view = MakeWindowTensorView(tensor_type, parent_shape, window_shape);
  if (!new_view.has_value()) return nullptr;
  return std::make_shared<TensorType>(window_shape, tensor_type->dtype_, tensor_type->memref_, new_view);
}

std::vector<ExprPtr> SubstituteExprVector(const std::vector<ExprPtr>& exprs,
                                          const std::unordered_map<const Var*, ExprPtr>& subst) {
  std::vector<ExprPtr> result;
  result.reserve(exprs.size());
  for (const auto& expr : exprs) {
    result.push_back(transform_utils::Substitute(expr, subst));
  }
  return result;
}

bool ExprVectorsPointerEqual(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].get() != rhs[i].get()) return false;
  }
  return true;
}

TypePtr SubstituteTypeExprs(const TypePtr& type, const std::unordered_map<const Var*, ExprPtr>& subst) {
  if (!type || subst.empty()) return type;
  if (auto tuple_type = As<TupleType>(type)) {
    std::vector<TypePtr> new_types;
    new_types.reserve(tuple_type->types_.size());
    bool changed = false;
    for (const auto& elem_type : tuple_type->types_) {
      auto new_type = SubstituteTypeExprs(elem_type, subst);
      changed = changed || new_type.get() != elem_type.get();
      new_types.push_back(std::move(new_type));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(new_types));
  }
  if (auto tensor_type = As<TensorType>(type)) {
    auto new_shape = SubstituteExprVector(tensor_type->shape_, subst);
    auto new_view = tensor_type->tensor_view_;
    if (new_view.has_value()) {
      new_view->stride = SubstituteExprVector(new_view->stride, subst);
      new_view->valid_shape = SubstituteExprVector(new_view->valid_shape, subst);
    }
    const bool shape_changed = !ExprVectorsPointerEqual(new_shape, tensor_type->shape_);
    bool view_changed = false;
    if (new_view.has_value() != tensor_type->tensor_view_.has_value()) {
      view_changed = true;
    } else if (new_view.has_value()) {
      view_changed = !ExprVectorsPointerEqual(new_view->stride, tensor_type->tensor_view_->stride) ||
                     !ExprVectorsPointerEqual(new_view->valid_shape, tensor_type->tensor_view_->valid_shape);
    }
    if (!shape_changed && !view_changed) return type;
    return std::make_shared<TensorType>(std::move(new_shape), tensor_type->dtype_, tensor_type->memref_,
                                        std::move(new_view));
  }
  return type;
}

std::optional<int64_t> CheckedAdd(int64_t lhs, int64_t rhs) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
    return std::nullopt;
  }
  return lhs + rhs;
}

std::optional<int64_t> CheckedSub(int64_t lhs, int64_t rhs) {
  if (rhs == std::numeric_limits<int64_t>::min()) {
    return std::nullopt;
  }
  return CheckedAdd(lhs, -rhs);
}

std::optional<int64_t> CheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs == 0 || rhs == 0) return int64_t{0};
  if (lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) return std::nullopt;
  if (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()) return std::nullopt;
  if (lhs > 0) {
    if (rhs > 0) {
      if (lhs > std::numeric_limits<int64_t>::max() / rhs) return std::nullopt;
    } else if (rhs < std::numeric_limits<int64_t>::min() / lhs) {
      return std::nullopt;
    }
  } else {
    if (rhs > 0) {
      if (lhs < std::numeric_limits<int64_t>::min() / rhs) return std::nullopt;
    } else if (lhs < std::numeric_limits<int64_t>::max() / rhs) {
      return std::nullopt;
    }
  }
  return lhs * rhs;
}

std::optional<int64_t> CheckedAbs(int64_t value) {
  if (value == std::numeric_limits<int64_t>::min()) return std::nullopt;
  return value < 0 ? -value : value;
}

bool AddLinearCoeff(LinearIndexExpr* expr, const Var* var, int64_t coeff) {
  if (!expr || !var || coeff == 0) return true;
  auto& slot = expr->coeffs[var];
  auto sum = CheckedAdd(slot, coeff);
  if (!sum.has_value()) return false;
  slot = *sum;
  if (slot == 0) expr->coeffs.erase(var);
  return true;
}

std::optional<LinearIndexExpr> ParseLinearIndexExpr(const ExprPtr& expr) {
  if (!expr) return std::nullopt;
  if (auto ci = As<ConstInt>(expr)) {
    return LinearIndexExpr{{}, ci->value_};
  }
  if (auto var = AsVarLike(expr)) {
    LinearIndexExpr result;
    AddLinearCoeff(&result, var.get(), 1);
    return result;
  }
  if (auto add = As<Add>(expr)) {
    auto lhs = ParseLinearIndexExpr(add->left_);
    auto rhs = ParseLinearIndexExpr(add->right_);
    if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
    auto constant = CheckedAdd(lhs->constant, rhs->constant);
    if (!constant.has_value()) return std::nullopt;
    lhs->constant = *constant;
    for (const auto& [var, coeff] : rhs->coeffs) {
      if (!AddLinearCoeff(&*lhs, var, coeff)) return std::nullopt;
    }
    return lhs;
  }
  if (auto sub = As<Sub>(expr)) {
    auto lhs = ParseLinearIndexExpr(sub->left_);
    auto rhs = ParseLinearIndexExpr(sub->right_);
    if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
    auto constant = CheckedSub(lhs->constant, rhs->constant);
    if (!constant.has_value()) return std::nullopt;
    lhs->constant = *constant;
    for (const auto& [var, coeff] : rhs->coeffs) {
      auto neg_coeff = CheckedSub(0, coeff);
      if (!neg_coeff.has_value()) return std::nullopt;
      if (!AddLinearCoeff(&*lhs, var, *neg_coeff)) return std::nullopt;
    }
    return lhs;
  }
  if (auto mul = As<Mul>(expr)) {
    auto lhs_ci = As<ConstInt>(mul->left_);
    auto rhs_ci = As<ConstInt>(mul->right_);
    ExprPtr scaled_expr;
    int64_t scale = 0;
    if (lhs_ci) {
      scaled_expr = mul->right_;
      scale = lhs_ci->value_;
    } else if (rhs_ci) {
      scaled_expr = mul->left_;
      scale = rhs_ci->value_;
    } else {
      return std::nullopt;
    }
    auto parsed = ParseLinearIndexExpr(scaled_expr);
    if (!parsed.has_value()) return std::nullopt;
    auto constant = CheckedMul(parsed->constant, scale);
    if (!constant.has_value()) return std::nullopt;
    parsed->constant = *constant;
    std::vector<const Var*> zero_coeff_vars;
    for (auto& [var, coeff] : parsed->coeffs) {
      auto scaled_coeff = CheckedMul(coeff, scale);
      if (!scaled_coeff.has_value()) return std::nullopt;
      coeff = *scaled_coeff;
      if (coeff == 0) zero_coeff_vars.push_back(var);
    }
    for (const auto* var : zero_coeff_vars) parsed->coeffs.erase(var);
    return parsed;
  }
  return std::nullopt;
}

std::optional<int64_t> ConstantDiffIfSameLinearBase(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto lhs_linear = ParseLinearIndexExpr(lhs);
  auto rhs_linear = ParseLinearIndexExpr(rhs);
  if (!lhs_linear.has_value() || !rhs_linear.has_value()) return std::nullopt;
  if (lhs_linear->coeffs != rhs_linear->coeffs) return std::nullopt;
  return CheckedSub(lhs_linear->constant, rhs_linear->constant);
}

std::optional<AffineForm> ParseAffineInLoop(const ExprPtr& expr, const Var* loop_var) {
  if (!expr) return std::nullopt;
  if (CountVarRefsInExpr(expr, loop_var) == 0) {
    return AffineForm{0, expr};
  }
  if (auto ci = As<ConstInt>(expr)) {
    return AffineForm{0, expr};
  }
  if (auto var = AsVarLike(expr)) {
    if (var.get() == loop_var) {
      auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, expr->span_);
      return AffineForm{1, zero};
    }
    return AffineForm{0, expr};
  }
  if (auto add = As<Add>(expr)) {
    auto lhs = ParseAffineInLoop(add->left_, loop_var);
    auto rhs = ParseAffineInLoop(add->right_, loop_var);
    if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
    auto coeff = CheckedAdd(lhs->coeff, rhs->coeff);
    if (!coeff.has_value()) return std::nullopt;
    return AffineForm{*coeff, MakeAdd(lhs->base, rhs->base, expr->span_)};
  }
  if (auto sub = As<Sub>(expr)) {
    auto lhs = ParseAffineInLoop(sub->left_, loop_var);
    auto rhs = ParseAffineInLoop(sub->right_, loop_var);
    if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
    auto coeff = CheckedSub(lhs->coeff, rhs->coeff);
    if (!coeff.has_value()) return std::nullopt;
    return AffineForm{*coeff, MakeSub(lhs->base, rhs->base, expr->span_)};
  }
  if (auto mul = As<Mul>(expr)) {
    auto lhs_ci = As<ConstInt>(mul->left_);
    auto rhs_ci = As<ConstInt>(mul->right_);
    if (lhs_ci) {
      auto rhs = ParseAffineInLoop(mul->right_, loop_var);
      if (!rhs.has_value()) return std::nullopt;
      auto coeff = CheckedMul(lhs_ci->value_, rhs->coeff);
      if (!coeff.has_value()) return std::nullopt;
      return AffineForm{*coeff,
                        MakeMul(std::make_shared<ConstInt>(lhs_ci->value_, lhs_ci->dtype(), lhs_ci->span_),
                                rhs->base, expr->span_)};
    }
    if (rhs_ci) {
      auto lhs = ParseAffineInLoop(mul->left_, loop_var);
      if (!lhs.has_value()) return std::nullopt;
      auto coeff = CheckedMul(rhs_ci->value_, lhs->coeff);
      if (!coeff.has_value()) return std::nullopt;
      return AffineForm{
          *coeff,
          MakeMul(lhs->base, std::make_shared<ConstInt>(rhs_ci->value_, rhs_ci->dtype(), rhs_ci->span_),
                  expr->span_)};
    }
  }
  return std::nullopt;
}

std::optional<int64_t> GetStaticTripCount(const ForStmtPtr& loop) {
  if (!loop) return std::nullopt;
  auto start = transform_utils::EvalConstInt(loop->start_);
  auto stop = transform_utils::EvalConstInt(loop->stop_);
  auto step = transform_utils::EvalConstInt(loop->step_);
  if (!start.has_value() || !stop.has_value() || !step.has_value() || *step == 0) return std::nullopt;
  if ((*step > 0 && *stop <= *start) || (*step < 0 && *stop >= *start)) return int64_t{0};
  auto distance = CheckedSub(*stop, *start);
  if (!distance.has_value()) return std::nullopt;
  auto step_abs = CheckedAbs(*step);
  auto distance_abs = CheckedAbs(*distance);
  if (!step_abs.has_value() || !distance_abs.has_value()) return std::nullopt;
  return CheckedCeilDiv(*distance_abs, *step_abs);
}

std::optional<int64_t> GetKnownPositiveTripCount(const ForStmtPtr& loop) {
  auto static_trip_count = GetStaticTripCount(loop);
  if (static_trip_count.has_value()) return static_trip_count;
  if (!loop) return std::nullopt;
  auto step = transform_utils::EvalConstInt(loop->step_);
  if (!step.has_value() || *step == 0) return std::nullopt;

  auto distance_expr = *step > 0 ? MakeSub(loop->stop_, loop->start_, loop->span_)
                                 : MakeSub(loop->start_, loop->stop_, loop->span_);
  distance_expr = arith::Analyzer().Simplify(distance_expr);
  auto distance = As<ConstInt>(distance_expr);
  int64_t distance_value = 0;
  if (distance) {
    distance_value = distance->value_;
  } else {
    auto linear_distance = *step > 0 ? ConstantDiffIfSameLinearBase(loop->stop_, loop->start_)
                                     : ConstantDiffIfSameLinearBase(loop->start_, loop->stop_);
    if (!linear_distance.has_value()) return std::nullopt;
    distance_value = *linear_distance;
  }
  if (distance_value <= 0) return int64_t{0};
  auto step_abs = CheckedAbs(*step);
  if (!step_abs.has_value()) return std::nullopt;
  return CheckedCeilDiv(distance_value, *step_abs);
}

std::optional<ExprPtr> SimplifyWithLoopBound(const ExprPtr& expr, const VarPtr& loop_var, int64_t value) {
  if (!expr) return std::nullopt;
  arith::Analyzer analyzer;
  analyzer.Bind(loop_var, value, value + 1);
  return analyzer.Simplify(expr);
}

std::optional<ExprPtr> SimplifyWithLoopValue(const ExprPtr& expr, const VarPtr& loop_var,
                                             const ExprPtr& value) {
  if (!expr || !value) return std::nullopt;
  arith::Analyzer analyzer;
  analyzer.Bind(loop_var, value);
  return analyzer.Simplify(expr);
}

std::optional<ExprPtr> GetLoopValueAtTrip(const ForStmtPtr& loop, int64_t trip_index) {
  if (!loop || trip_index < 0) return std::nullopt;
  auto step = transform_utils::EvalConstInt(loop->step_);
  if (!step.has_value()) return std::nullopt;
  auto delta = CheckedMul(trip_index, *step);
  if (!delta.has_value()) return std::nullopt;
  if (*delta == 0) return loop->start_;
  auto delta_expr = std::make_shared<ConstInt>(*delta, DataType::INDEX, loop->span_);
  return arith::Analyzer().Simplify(MakeAdd(loop->start_, delta_expr, loop->span_));
}
}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto
