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

/**
 * @file block_mx_scale_tensor_views_pass.cpp
 * @brief BlockMxScaleTensorViews pass — physicalize logical MX scale tensor views.
 *
 * MX scale tensors use logical rank-2 shapes in the DSL, but A5 TLoadMxCube*
 * consumes packed rank-5 SFractal GlobalTensor views:
 *
 *   MX_A_ZZ [M, G] -> [1, M/16, G/2, 16, 2]
 *   MX_B_NN [G, N] -> [1, N/16, G/2, 16, 2]
 *
 * This pass owns that conversion, including tensor types, tile.load windows,
 * ND/MX backing aliases, Call/Submit result types, and symbolic offset proofs.
 * It runs after BlockNzTensorViews and before MaterializeTensorStrides.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kMxBlockedAttr = "mx_tensor_views_blocked";
constexpr int kMxProofStepBudget = 256;

struct MxOffsetFacts {
  std::function<ExprPtr(const VarPtr&)> definition;
  std::function<bool(const VarPtr&, int64_t, int*)> is_multiple_of;
  std::function<bool(const VarPtr&, int*)> is_non_negative;
};

bool IsMxProvableMultipleOf(const ExprPtr& expr, int64_t divisor, const MxOffsetFacts& facts, int* budget) {
  INTERNAL_CHECK(divisor > 0 && (divisor & (divisor - 1)) == 0)
      << "Internal error: MX divisibility proofs require a power-of-two divisor, got " << divisor;
  if (!expr || --*budget < 0) return false;

  if (auto value = As<ConstInt>(expr)) return value->value_ % divisor == 0;
  if (auto mul = As<Mul>(expr)) {
    return IsMxProvableMultipleOf(mul->left_, divisor, facts, budget) ||
           IsMxProvableMultipleOf(mul->right_, divisor, facts, budget);
  }
  if (auto add = As<Add>(expr)) {
    return IsMxProvableMultipleOf(add->left_, divisor, facts, budget) &&
           IsMxProvableMultipleOf(add->right_, divisor, facts, budget);
  }
  if (auto sub = As<Sub>(expr)) {
    return IsMxProvableMultipleOf(sub->left_, divisor, facts, budget) &&
           IsMxProvableMultipleOf(sub->right_, divisor, facts, budget);
  }
  if (auto floordiv = As<FloorDiv>(expr)) {
    auto denominator = As<ConstInt>(floordiv->right_);
    if (!denominator || denominator->value_ <= 0 || (denominator->value_ & (denominator->value_ - 1)) != 0) {
      return false;
    }
    int64_t combined = 0;
    if (__builtin_mul_overflow(denominator->value_, divisor, &combined)) return false;
    return IsMxProvableMultipleOf(floordiv->left_, combined, facts, budget);
  }
  if (auto var = As<Var>(expr)) {
    if (facts.is_multiple_of && facts.is_multiple_of(var, divisor, budget)) return true;
    if (facts.definition) {
      if (auto definition = facts.definition(var)) {
        return IsMxProvableMultipleOf(definition, divisor, facts, budget);
      }
    }
  }
  return false;
}

bool IsMxProvableNonNegative(const ExprPtr& expr, const MxOffsetFacts& facts, int* budget) {
  if (!expr || --*budget < 0) return false;

  if (auto value = As<ConstInt>(expr)) return value->value_ >= 0;
  if (auto mul = As<Mul>(expr)) {
    return IsMxProvableNonNegative(mul->left_, facts, budget) &&
           IsMxProvableNonNegative(mul->right_, facts, budget);
  }
  if (auto add = As<Add>(expr)) {
    return IsMxProvableNonNegative(add->left_, facts, budget) &&
           IsMxProvableNonNegative(add->right_, facts, budget);
  }
  if (auto floordiv = As<FloorDiv>(expr)) {
    auto denominator = As<ConstInt>(floordiv->right_);
    if (!denominator || denominator->value_ <= 0) return false;
    return IsMxProvableNonNegative(floordiv->left_, facts, budget);
  }
  if (auto var = As<Var>(expr)) {
    if (facts.is_non_negative && facts.is_non_negative(var, budget)) return true;
    if (facts.definition) {
      if (auto definition = facts.definition(var)) {
        return IsMxProvableNonNegative(definition, facts, budget);
      }
    }
  }
  return false;
}

bool IsMxTensorType(const TypePtr& type) {
  auto tensor_type = AsTensorTypeLike(type);
  if (!tensor_type) return false;
  const auto& view = tensor_type->tensor_view_;
  return view.has_value() && IsMxTensorLayout(view->layout);
}

std::vector<ExprPtr> BlockMxShape(const std::vector<ExprPtr>& shape, TensorLayout layout, const Span& span) {
  CHECK_SPAN(IsMxTensorLayout(layout), span)
      << "BlockMxShape requires MX_A_ZZ or MX_B_NN, got " << TensorLayoutToString(layout);
  CHECK_SPAN(shape.size() == 2, span)
      << "MX layout requires a rank-2 logical shape, got rank " << shape.size();

  const bool is_a = layout == TensorLayout::MX_A_ZZ;
  const size_t block_axis = is_a ? 0 : 1;
  const size_t group_axis = is_a ? 1 : 0;
  constexpr int64_t kRows = tile_view_semantics::kMXSFractalRows;
  constexpr int64_t kCols = tile_view_semantics::kMXSFractalCols;
  const std::string layout_name = TensorLayoutToString(layout);
  auto block = As<ConstInt>(shape[block_axis]);
  auto group = As<ConstInt>(shape[group_axis]);
  CHECK_SPAN(block, span) << layout_name << " layout requires a static block dimension, got a dynamic "
                          << "extent. Dynamic MX tensors are not supported yet.";
  CHECK_SPAN(group, span) << layout_name << " layout requires a static group dimension, got a dynamic "
                          << "extent. Dynamic MX tensors are not supported yet.";
  CHECK_SPAN(block->value_ > 0 && block->value_ % kRows == 0, span)
      << layout_name << " layout requires the block dimension to be a positive multiple of " << kRows
      << ", got " << block->value_ << ".";
  CHECK_SPAN(group->value_ > 0 && group->value_ % kCols == 0, span)
      << layout_name << " layout requires the group dimension to be a positive multiple of " << kCols
      << ", got " << group->value_ << ".";

  auto make_index = [&span](int64_t value) {
    return std::make_shared<ConstInt>(value, DataType::INDEX, span);
  };
  return {make_index(1), make_index(block->value_ / kRows), make_index(group->value_ / kCols),
          make_index(kRows), make_index(kCols)};
}

std::vector<ExprPtr> BlockMxOffsets(const std::vector<ExprPtr>& offsets, TensorLayout layout,
                                    const Span& span, const MxOffsetFacts& facts) {
  CHECK_SPAN(IsMxTensorLayout(layout), span)
      << "BlockMxOffsets requires MX_A_ZZ or MX_B_NN, got " << TensorLayoutToString(layout);
  CHECK_SPAN(offsets.size() == 2, span) << "MX layout requires rank-2 offsets, got " << offsets.size();

  const bool is_a = layout == TensorLayout::MX_A_ZZ;
  const size_t block_axis = is_a ? 0 : 1;
  const size_t group_axis = is_a ? 1 : 0;
  constexpr int64_t kRows = tile_view_semantics::kMXSFractalRows;
  constexpr int64_t kCols = tile_view_semantics::kMXSFractalCols;
  const std::string layout_name = TensorLayoutToString(layout);

  auto block_axis_offset = [&](const ExprPtr& offset, int64_t divisor, const char* axis_name) -> ExprPtr {
    if (auto value = As<ConstInt>(offset)) {
      CHECK_SPAN(value->value_ >= 0 && value->value_ % divisor == 0, span)
          << layout_name << " slice offset on the " << axis_name
          << " axis must be a non-negative multiple of " << divisor << ", got " << value->value_ << ".";
      return std::make_shared<ConstInt>(value->value_ / divisor, DataType::INDEX, span);
    }
    int divisibility_budget = kMxProofStepBudget;
    CHECK_SPAN(IsMxProvableMultipleOf(offset, divisor, facts, &divisibility_budget), span)
        << layout_name << " layout requires the slice offset on the " << axis_name
        << " axis to be a multiple of " << divisor
        << ", and this one cannot be proven to be. Provable forms are a constant, a loop variable whose "
        << "start and step are both multiples, a floor-division with a suitably aligned numerator, and "
        << "sums, differences, or constant multiples built from those. Slice on an aligned boundary.";
    int sign_budget = kMxProofStepBudget;
    CHECK_SPAN(IsMxProvableNonNegative(offset, facts, &sign_budget), span)
        << layout_name << " layout requires the slice offset on the " << axis_name
        << " axis to be non-negative, and this one cannot be proven to be. A negative partition offset "
        << "would otherwise be clamped and silently read the wrong SFractal.";
    return MakeFloorDiv(offset, std::make_shared<ConstInt>(divisor, DataType::INDEX, span), span);
  };

  auto block_offset = block_axis_offset(offsets[block_axis], kRows, "block");
  auto group_offset = block_axis_offset(offsets[group_axis], kCols, "group");
  auto make_index = [&span](int64_t value) {
    return std::make_shared<ConstInt>(value, DataType::INDEX, span);
  };
  return {make_index(0), std::move(block_offset), std::move(group_offset), make_index(0), make_index(0)};
}

TypePtr BlockMxType(const TypePtr& type, const Span& span) {
  if (!type) return type;
  if (auto tuple_type = As<TupleType>(type)) {
    std::vector<TypePtr> new_elements;
    new_elements.reserve(tuple_type->types_.size());
    bool changed = false;
    for (const auto& element : tuple_type->types_) {
      auto new_element = BlockMxType(element, span);
      if (new_element.get() != element.get()) changed = true;
      new_elements.push_back(std::move(new_element));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(new_elements));
  }
  if (As<DistributedTensorType>(type)) {
    if (!IsMxTensorType(type)) return type;
    CHECK_SPAN(false, span) << "MX layout is not supported on a distributed tensor yet. "
                            << "Annotate the tensor as pl.ND.";
  }
  if (auto tensor_type = As<TensorType>(type)) {
    if (!IsMxTensorType(type)) return type;
    const auto& maybe_view = tensor_type->tensor_view_;
    if (!maybe_view.has_value()) return type;
    const TensorView& view = *maybe_view;
    auto blocked_shape = BlockMxShape(tensor_type->shape_, view.layout, span);
    CHECK_SPAN(view.valid_shape.empty(), span)
        << "MX layout does not support a partial tensor valid_shape yet; the whole tensor must be valid.";
    TensorView blocked_view(/*stride=*/{}, view.layout, view.valid_shape, view.pad);
    return std::make_shared<TensorType>(std::move(blocked_shape), tensor_type->dtype_, tensor_type->memref_,
                                        blocked_view);
  }
  return type;
}

class MxOffsetFactStore {
 public:
  explicit MxOffsetFactStore(ProgramPtr program) : program_(std::move(program)) {
    if (!program_) return;
    Collector collector(this);
    collector.VisitProgram(program_);
  }

  [[nodiscard]] MxOffsetFacts Facts() const {
    MxOffsetFacts facts;
    facts.definition = [this](const VarPtr& var) -> ExprPtr {
      auto it = definitions_.find(var);
      return it == definitions_.end() ? nullptr : it->second;
    };
    facts.is_multiple_of = [this](const VarPtr& var, int64_t divisor, int* budget) {
      auto loop = loop_bindings_.find(var);
      if (loop != loop_bindings_.end()) {
        const auto& [start, step] = loop->second;
        if (start % divisor == 0 && step % divisor == 0) return true;
      }
      auto sources = parameter_sources_.find(var);
      if (sources == parameter_sources_.end() || sources->second.empty()) return false;
      auto recursive_facts = Facts();
      for (const auto& source : sources->second) {
        if (!IsMxProvableMultipleOf(source, divisor, recursive_facts, budget)) return false;
      }
      return true;
    };
    facts.is_non_negative = [this](const VarPtr& var, int* budget) {
      if (non_negative_vars_.count(var) != 0) return true;
      auto loop = loop_bindings_.find(var);
      if (loop != loop_bindings_.end()) {
        const auto& [start, step] = loop->second;
        if (start >= 0 && step >= 0) return true;
      }
      auto sources = parameter_sources_.find(var);
      if (sources == parameter_sources_.end() || sources->second.empty()) return false;
      auto recursive_facts = Facts();
      for (const auto& source : sources->second) {
        if (!IsMxProvableNonNegative(source, recursive_facts, budget)) return false;
      }
      return true;
    };
    return facts;
  }

 private:
  void RecordParamSource(const VarPtr& param, const ExprPtr& source) {
    if (param && As<ScalarType>(param->GetType())) parameter_sources_[param].push_back(source);
  }

  void RecordUnknownParamSources(const FunctionPtr& callee) {
    for (const auto& param : callee->params_) RecordParamSource(param, nullptr);
  }

  void RecordCall(const OpPtr& op, const std::vector<ExprPtr>& args, bool is_submit) {
    auto callee_var = As<GlobalVar>(op);
    if (!callee_var || !program_) return;
    auto callee = program_->GetFunction(callee_var->name_);
    if (!callee) return;

    if (!is_submit) {
      if (args.size() != callee->params_.size()) {
        RecordUnknownParamSources(callee);
        return;
      }
      for (size_t i = 0; i < args.size(); ++i) RecordParamSource(callee->params_[i], args[i]);
      return;
    }

    if (args.size() > callee->params_.size()) {
      RecordUnknownParamSources(callee);
      return;
    }
    size_t ctx_count = 0;
    while (ctx_count < callee->params_.size() &&
           IsA<CommCtxType>(callee->params_[callee->params_.size() - 1 - ctx_count]->GetType())) {
      ++ctx_count;
    }
    if (args.size() < ctx_count) {
      RecordUnknownParamSources(callee);
      return;
    }
    const size_t prefix_count = args.size() - ctx_count;
    const size_t gap = callee->params_.size() - args.size();
    for (size_t i = 0; i < prefix_count; ++i) RecordParamSource(callee->params_[i], args[i]);
    for (size_t i = prefix_count; i < args.size(); ++i) {
      RecordParamSource(callee->params_[i + gap], args[i]);
    }
  }

  class Collector : public IRVisitor {
   public:
    explicit Collector(MxOffsetFactStore* store) : store_(store) {}

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      if (op->var_ && As<ScalarType>(op->var_->GetType())) {
        store_->definitions_.emplace(op->var_, op->value_);
        auto call = As<Call>(op->value_);
        if (call && (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num"))) {
          store_->non_negative_vars_.insert(op->var_);
        }
      }
      IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForStmtPtr& op) override {
      auto start = As<ConstInt>(op->start_);
      auto step = As<ConstInt>(op->step_);
      if (op->loop_var_ && start && step) {
        store_->loop_bindings_.emplace(op->loop_var_, std::make_pair(start->value_, step->value_));
      }
      IRVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const CallPtr& op) override {
      store_->RecordCall(op->op_, op->args_, /*is_submit=*/false);
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const SubmitPtr& op) override {
      store_->RecordCall(op->op_, op->args_, /*is_submit=*/true);
      IRVisitor::VisitExpr_(op);
    }

   private:
    MxOffsetFactStore* store_;
  };

  ProgramPtr program_;
  std::unordered_map<VarPtr, ExprPtr> definitions_;
  std::unordered_map<VarPtr, std::pair<int64_t, int64_t>> loop_bindings_;
  std::unordered_set<VarPtr> non_negative_vars_;
  std::unordered_map<VarPtr, std::vector<ExprPtr>> parameter_sources_;
};

ExprPtr BlockMxTupleArg(const ExprPtr& arg, TensorLayout layout, const Span& span, bool is_offsets,
                        const MxOffsetFacts& facts) {
  auto tuple = As<MakeTuple>(arg);
  INTERNAL_CHECK_SPAN(tuple, span) << "Internal error: tile.load coordinate argument must be a MakeTuple";
  auto blocked = is_offsets ? BlockMxOffsets(tuple->elements_, layout, span, facts)
                            : BlockMxShape(tuple->elements_, layout, span);
  return std::make_shared<MakeTuple>(std::move(blocked), tuple->span_);
}

class BlockMxMutator : public IRMutator {
 public:
  explicit BlockMxMutator(MxOffsetFacts facts) : facts_(std::move(facts)) {}

  void AddSubstitution(const VarPtr& old_var, const VarPtr& new_var) { var_cache_[old_var] = new_var; }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto cached = var_cache_.find(op);
    if (cached != var_cache_.end()) return cached->second;
    auto new_type = BlockMxType(op->GetType(), op->span_);
    if (new_type.get() == op->GetType().get()) {
      var_cache_[op] = op;
      return op;
    }
    auto new_var = std::make_shared<Var>(op->name_hint_, std::move(new_type), op->span_);
    var_cache_[op] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto cached = var_cache_.find(op);
    if (cached != var_cache_.end()) return cached->second;
    auto new_init = IRMutator::VisitExpr(op->initValue_);
    auto new_type = BlockMxType(op->GetType(), op->span_);
    if (new_init.get() == op->initValue_.get() && new_type.get() == op->GetType().get()) {
      var_cache_[op] = op;
      return op;
    }
    auto new_iter_arg = std::make_shared<IterArg>(op->name_hint_, std::move(new_type), new_init, op->span_);
    var_cache_[op] = new_iter_arg;
    return new_iter_arg;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    std::vector<ExprPtr> new_args;
    new_args.reserve(op->args_.size());
    bool args_changed = false;
    for (const auto& arg : op->args_) {
      auto new_arg = IRMutator::VisitExpr(arg);
      if (new_arg.get() != arg.get()) args_changed = true;
      new_args.push_back(std::move(new_arg));
    }

    std::vector<size_t> mx_args;
    for (size_t i = 0; i < new_args.size(); ++i) {
      auto tensor = AsVarLike(new_args[i]);
      if (tensor && IsMxTensorType(tensor->GetType())) mx_args.push_back(i);
    }

    const bool is_function_call = static_cast<bool>(As<GlobalVar>(op->op_));
    if (!mx_args.empty() && !is_function_call) {
      CHECK_SPAN(!IsOp(op, "tile.store"), op->span_)
          << "MX layout is read-only: an MX scale tensor cannot be a store destination.";
      const bool is_load_source = IsOp(op, "tile.load") && mx_args.size() == 1 && mx_args[0] == 0;
      const bool is_backing_alias = IsOp(op, "tensor.view") && mx_args.size() == 1 && mx_args[0] == 0;
      CHECK_SPAN(is_load_source || is_backing_alias, op->span_)
          << "MX layout currently supports only 'tile.load' and shaped FP8E8M0 'tensor.view' backing "
          << "aliases reading the tensor as their source, but it is used by '" << op->op_->name_
          << "' at argument " << mx_args[0] << ".";
      if (is_load_source) {
        new_args = BlockMxTileLoadArgs(op, std::move(new_args));
        args_changed = true;
      }
    }

    if (IsOp(op, "tensor.view") && IsMxTensorType(op->GetType())) {
      auto output_type = As<TensorType>(op->GetType());
      INTERNAL_CHECK_SPAN(output_type, op->span_)
          << "Internal error: an MX tensor.view result must carry a TensorType";
      const auto& maybe_view = output_type->tensor_view_;
      INTERNAL_CHECK_SPAN(maybe_view.has_value(), op->span_)
          << "Internal error: an MX tensor.view result must carry a TensorView";
      INTERNAL_CHECK_SPAN(new_args.size() >= 2, op->span_)
          << "Internal error: an ND-to-MX tensor.view backing alias must carry a shape argument";
      new_args[1] = BlockMxTupleArg(new_args[1], maybe_view->layout, op->span_, /*is_offsets=*/false, facts_);
      args_changed = true;
    }

    auto new_return_type = BlockMxType(op->GetType(), op->span_);
    std::vector<std::pair<std::string, std::any>> new_attrs;
    bool attrs_changed = false;
    new_attrs.reserve(op->attrs_.size());
    for (const auto& [key, value] : op->attrs_) {
      auto remapped = MapAttrExprs(value, [this](const ExprPtr& expr) { return IRMutator::VisitExpr(expr); });
      if (remapped.has_value()) {
        attrs_changed = true;
        new_attrs.emplace_back(key, std::move(*remapped));
      } else {
        new_attrs.emplace_back(key, value);
      }
    }

    if (!args_changed && new_return_type.get() == op->GetType().get() && !attrs_changed) return op;
    std::vector<std::pair<std::string, std::any>> attrs_to_use;
    if (attrs_changed) {
      attrs_to_use = std::move(new_attrs);
    } else {
      attrs_to_use = op->attrs_;
    }
    return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, std::move(attrs_to_use),
                                  std::move(new_return_type), op->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto submit = As<Submit>(base);
    INTERNAL_CHECK_SPAN(submit, op->span_)
        << "Internal error: BlockMxScaleTensorViews visited a Submit to a non-Submit expression";
    auto new_return_type = BlockMxType(submit->GetType(), submit->span_);
    if (new_return_type.get() == submit->GetType().get()) return submit;
    return std::make_shared<Submit>(submit->op_, submit->args_, submit->deps_, submit->kwargs_,
                                    submit->attrs_, std::move(new_return_type), submit->span_,
                                    submit->core_num_, submit->sync_start_, submit->allow_early_resolve_,
                                    submit->predicate_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_var_expr = IRMutator::VisitExpr(op->var_);
    auto new_value = IRMutator::VisitExpr(op->value_);
    auto new_var = As<Var>(new_var_expr);
    INTERNAL_CHECK(new_var)
        << "Internal error: BlockMxScaleTensorViews visited an AssignStmt LHS to a non-Var";
    if (new_var.get() == op->var_.get() && new_value.get() == op->value_.get()) return op;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

 private:
  std::vector<ExprPtr> BlockMxTileLoadArgs(const CallPtr& op, std::vector<ExprPtr> args) {
    INTERNAL_CHECK_SPAN(args.size() >= 3, op->span_)
        << "Internal error: tile.load expects at least (tensor, offsets, shapes), got " << args.size();
    auto tensor = AsVarLike(args[0]);
    auto tensor_type = AsTensorTypeLike(tensor->GetType());
    INTERNAL_CHECK_SPAN(tensor_type, op->span_)
        << "Internal error: MX tile.load source must carry a TensorType";
    const auto& maybe_view = tensor_type->tensor_view_;
    INTERNAL_CHECK_SPAN(maybe_view.has_value(), op->span_)
        << "Internal error: MX tile.load source must carry a TensorView";
    const TensorLayout layout = maybe_view->layout;

    std::optional<MemorySpace> target;
    for (const auto& [key, value] : op->kwargs_) {
      if (key == "target_memory") {
        target = AnyCast<MemorySpace>(value, "target_memory");
        break;
      }
    }
    CHECK_SPAN(target.has_value() && *target == MemorySpace::Mat, op->span_)
        << "MX layout currently supports only cube scale loads (target_memory=pl.Mem.Mat), got "
        << (target.has_value() ? MemorySpaceToString(*target) : std::string("no target_memory")) << ".";

    args[1] = BlockMxTupleArg(args[1], layout, op->span_, /*is_offsets=*/true, facts_);
    args[2] = BlockMxTupleArg(args[2], layout, op->span_, /*is_offsets=*/false, facts_);
    if (args.size() >= 4) args[3] = args[2];
    return args;
  }

  MxOffsetFacts facts_;
  std::unordered_map<VarPtr, VarPtr> var_cache_;
};

FunctionPtr TransformFunction(const FunctionPtr& func, const MxOffsetFacts& facts) {
  if (func->HasAttr(kMxBlockedAttr)) return func;

  bool params_changed = false;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  std::unordered_map<VarPtr, VarPtr> substitutions;
  for (const auto& old_param : func->params_) {
    auto new_type = BlockMxType(old_param->GetType(), old_param->span_);
    if (new_type.get() == old_param->GetType().get()) {
      new_params.push_back(old_param);
      continue;
    }
    auto new_param = std::make_shared<Var>(old_param->name_hint_, std::move(new_type), old_param->span_);
    new_params.push_back(new_param);
    substitutions.emplace(old_param, new_param);
    params_changed = true;
  }

  bool returns_changed = false;
  std::vector<TypePtr> new_return_types;
  new_return_types.reserve(func->return_types_.size());
  for (const auto& return_type : func->return_types_) {
    auto new_return_type = BlockMxType(return_type, func->span_);
    if (new_return_type.get() != return_type.get()) returns_changed = true;
    new_return_types.push_back(std::move(new_return_type));
  }

  BlockMxMutator mutator(facts);
  for (const auto& [old_var, new_var] : substitutions) mutator.AddSubstitution(old_var, new_var);
  StmtPtr new_body = func->body_;
  if (func->body_) new_body = mutator.VisitStmt(func->body_);
  const bool body_changed = new_body.get() != func->body_.get();

  auto new_func = MutableCopy(func);
  if (params_changed) new_func->params_ = std::move(new_params);
  if (returns_changed) new_func->return_types_ = std::move(new_return_types);
  if (body_changed) new_func->body_ = std::move(new_body);
  new_func->attrs_.emplace_back(kMxBlockedAttr, std::any(true));
  return new_func;
}

}  // namespace

namespace pass {

Pass BlockMxScaleTensorViews() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    MxOffsetFactStore fact_store(program);
    auto facts = fact_store.Facts();
    bool modified = false;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func, facts);
      if (new_func.get() != func.get()) modified = true;
      new_functions[gvar] = std::move(new_func);
    }
    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "BlockMxScaleTensorViews", kBlockMxScaleTensorViewsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
