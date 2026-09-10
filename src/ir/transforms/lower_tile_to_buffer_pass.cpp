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
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/property_verifier_registry.h"

namespace pypto {
namespace ir {
namespace {

std::vector<int64_t> StaticExtents(const std::vector<ExprPtr>& extents, const Span& span) {
  std::vector<int64_t> result;
  result.reserve(extents.size());
  for (const auto& extent : extents) {
    auto value = As<ConstInt>(extent);
    CHECK_SPAN(value, span) << "LowerTileToBuffer: this recipe requires static physical and valid extents";
    result.push_back(value->value_);
  }
  return result;
}

BufferTypePtr DenseDescriptor(const TileTypePtr& tile, const Span& span) {
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  CHECK_SPAN(tile->shape_.size() == 2 && tile->dtype_ == DataType::FP32 &&
                 tile->GetMemorySpace() == MemorySpace::Vec && view.blayout == TileLayout::row_major &&
                 view.slayout == TileLayout::none_box && view.fractal == 512 && view.pad == PadValue::null &&
                 view.compact == CompactMode::null,
             span)
      << "LowerTileToBuffer: this recipe requires dense rank-2 Vec FP32 tiles";
  const auto view_offset = As<ConstInt>(view.start_offset);
  CHECK_SPAN(!view.start_offset || (view_offset && view_offset->value_ == 0), span)
      << "LowerTileToBuffer: nonzero tile view offsets require an explicit Buffer view recipe";
  const auto shape = StaticExtents(tile->shape_, span);
  const auto valid = StaticExtents(view.valid_shape.empty() ? tile->shape_ : view.valid_shape, span);
  if (!view.stride.empty()) {
    const auto strides = StaticExtents(view.stride, span);
    CHECK_SPAN(strides == std::vector<int64_t>({shape[1], 1}), span)
        << "LowerTileToBuffer: strided tiles require an explicit Buffer view recipe";
  }
  return std::make_shared<BufferType>(shape, tile->dtype_, MemorySpace::Vec, valid);
}

/// One indexed allocation identity. No conversion table is attached to the IR.
struct BufferStorage {
  MemRefPtr memory;
  VarPtr handle;
};

class StorageIndex : public IRVisitor {
 public:
  explicit StorageIndex(bool addressed) : addressed_(addressed) {}

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    // Allocation passes attach planned storage to SSA variables. A producer
    // Call retains its logical deduced type and is not an allocation identity.
    auto variable = AsVarLike(expr);
    if (auto tile = variable ? As<TileType>(variable->GetType()) : nullptr;
        tile && types_.insert(tile.get()).second) {
      auto memory = GetDefinedMemRef(tile);
      CHECK_SPAN(memory && memory->base_ && !memory->is_pinned_ && memory->slot_count_ == 1 &&
                     !memory->slot_index_.has_value(),
                 expr->span_)
          << "LowerTileToBuffer: tile storage must be planned; multi-slot storage needs its own recipe";
      auto offset = As<ConstInt>(memory->byte_offset_);
      CHECK_SPAN(offset && offset->value_ >= 0 && (addressed_ || offset->value_ == 0), expr->span_)
          << "LowerTileToBuffer: expected a final nonnegative address or an addressless root window";
      auto descriptor = DenseDescriptor(tile, expr->span_);
      auto found = roots.find(memory->base_.get());
      if (found == roots.end()) {
        auto handle = std::make_shared<Var>(memory->base_->name_hint_ + "_buffer", descriptor, expr->span_);
        roots.emplace(memory->base_.get(), BufferStorage{memory, handle});
      } else {
        CHECK_SPAN(structural_equal(descriptor, found->second.handle->GetType()) &&
                       structural_equal(memory->byte_offset_, found->second.memory->byte_offset_),
                   expr->span_)
            << "LowerTileToBuffer: one allocation has differing descriptors or windows; "
               "an explicit Buffer view or dynamic-metadata recipe is required";
      }
    }
    IRVisitor::VisitExpr(expr);
  }

  std::unordered_map<const Var*, BufferStorage> roots;

 protected:
  // Initializers are visited once at their lexical binding. Body references
  // must not expand the initializer chains of enclosing loops.
  void VisitExpr_(const IterArgPtr& argument) override { VisitVarLike_(argument); }
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }

 private:
  bool addressed_;
  std::unordered_set<const TileType*> types_;
};

class TileToBufferMutator : public IRMutator {
 public:
  TileToBufferMutator(const StorageIndex& storage, bool addressed)
      : storage_(storage), addressed_(addressed) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& var) override {
    if (As<TileType>(var->GetType())) return Handle(var);
    auto alias = tensor_aliases_.find(var.get());
    return alias == tensor_aliases_.end() ? IRMutator::VisitExpr_(var) : alias->second;
  }

  ExprPtr VisitExpr_(const IterArgPtr& var) override {
    if (As<TileType>(var->GetType())) return Handle(var);
    if (auto alias = tensor_aliases_.find(var.get()); alias != tensor_aliases_.end()) {
      return alias->second;
    }
    auto found = var_remap_.find(var.get());
    INTERNAL_CHECK_SPAN(found != var_remap_.end(), var->span_)
        << "Internal error: Buffer conversion encountered an unbound scalar carry";
    return found->second;
  }

  ExprPtr VisitExpr_(const CallPtr& call) override {
    INTERNAL_CHECK_SPAN(call->op_, call->span_) << "Internal error: device call has no operator";
    CHECK_SPAN(false, call->span_) << "LowerTileToBuffer: no scalar or nested-call recipe for '"
                                   << call->op_->name_ << "'";
    return call;
  }

  ExprPtr VisitExpr_(const SubmitPtr& submit) override {
    CHECK_SPAN(false, submit->span_) << "LowerTileToBuffer: a device function cannot submit tasks";
    return submit;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto call = As<Call>(assign->value_)) return LowerCall(call, assign->var_);
    if (As<TileType>(assign->var_->GetType())) {
      auto source = AsVarLike(assign->value_);
      INTERNAL_CHECK_SPAN(source && Handle(source) == Handle(assign->var_), assign->span_)
          << "Internal error: Tile storage legalization left an implicit alias transfer";
      return Empty(assign->span_);
    }
    if (As<TensorType>(assign->var_->GetType())) {
      auto source = AsVarLike(VisitExpr(assign->value_));
      CHECK_SPAN(source, assign->span_)
          << "LowerTileToBuffer: GM assignments require a normalized tensor parameter alias";
      tensor_aliases_[assign->var_.get()] = source;
      return Empty(assign->span_);
    }
    return IRMutator::VisitStmt_(assign);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_)) return LowerCall(call, nullptr);
    return IRMutator::VisitStmt_(eval);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& branch) override {
    auto condition = VisitExpr(branch->condition_);
    YieldContext then_context(branch->return_vars_);
    auto then_body = LowerRegion(branch->then_body_, then_context);
    YieldContext else_context(branch->return_vars_);
    std::optional<StmtPtr> else_body;
    if (branch->else_body_) else_body = LowerRegion(*branch->else_body_, else_context);
    std::vector<VarPtr> results;
    for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
      const auto& result = branch->return_vars_[i];
      if (As<TileType>(result->GetType())) continue;
      if (As<TensorType>(result->GetType())) {
        CHECK_SPAN(
            then_context.tensor_values[i] && then_context.tensor_values[i] == else_context.tensor_values[i],
            branch->span_)
            << "LowerTileToBuffer: GM branch results must alias the same parameter in both arms";
        tensor_aliases_[result.get()] = then_context.tensor_values[i];
      } else {
        results.push_back(result);
      }
    }
    return std::make_shared<IfStmt>(condition, then_body, else_body, results, branch->span_,
                                    branch->leading_comments_);
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& yield) override {
    if (!yield_context_) return IRMutator::VisitStmt_(yield);
    INTERNAL_CHECK_SPAN(yield->value_.size() == yield_context_->results.size(), yield->span_)
        << "Internal error: device region yield/result arity mismatch";
    std::vector<ExprPtr> values;
    for (size_t i = 0; i < yield->value_.size(); ++i) {
      const auto& result = yield_context_->results[i];
      const auto& value = yield->value_[i];
      if (As<TileType>(result->GetType())) {
        INTERNAL_CHECK_SPAN(Handle(value) == Handle(result), yield->span_)
            << "Internal error: Tile storage legalization left an implicit region transfer";
      } else if (As<TensorType>(result->GetType())) {
        yield_context_->tensor_values[i] = VisitExpr(value);
      } else {
        values.push_back(VisitExpr(value));
      }
    }
    return std::make_shared<YieldStmt>(values, yield->span_, yield->leading_comments_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& loop) override { return LowerLoop(loop); }
  StmtPtr VisitStmt_(const WhileStmtPtr& loop) override { return LowerLoop(loop); }

 private:
  struct YieldContext {
    explicit YieldContext(const std::vector<VarPtr>& results)
        : results(results), tensor_values(results.size()) {}
    const std::vector<VarPtr>& results;
    std::vector<ExprPtr> tensor_values;
  };

  StmtPtr LowerRegion(const StmtPtr& body, YieldContext& context) {
    auto* outer = yield_context_;
    yield_context_ = &context;
    auto lowered = VisitStmt(body);
    yield_context_ = outer;
    return lowered;
  }

  template <typename LoopPtr>
  StmtPtr LowerLoop(const LoopPtr& loop) {
    INTERNAL_CHECK_SPAN(loop->iter_args_.size() == loop->return_vars_.size(), loop->span_)
        << "Internal error: device loop carry/result arity mismatch";
    auto lowered = std::make_shared<std::remove_const_t<typename LoopPtr::element_type>>(*loop);
    if constexpr (std::is_same_v<LoopPtr, ForStmtPtr>) {
      lowered->start_ = VisitExpr(loop->start_);
      lowered->stop_ = VisitExpr(loop->stop_);
      lowered->step_ = VisitExpr(loop->step_);
    }
    lowered->iter_args_.clear();
    lowered->return_vars_.clear();
    std::vector<ExprPtr> initial_values(loop->iter_args_.size());
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      const auto& result = loop->return_vars_[i];
      auto initial = VisitExpr(argument->initValue_);
      initial_values[i] = initial;
      if (As<TileType>(argument->GetType())) {
        INTERNAL_CHECK_SPAN(initial == Handle(argument) && initial == Handle(result), loop->span_)
            << "Internal error: Tile storage legalization left an implicit loop entry transfer";
      } else if (As<TensorType>(argument->GetType())) {
        tensor_aliases_[argument.get()] = initial;
      } else {
        CHECK_SPAN(As<ScalarType>(argument->GetType()), argument->span_)
            << "LowerTileToBuffer: loop carries require scalar, Tile, or normalized GM values";
        auto scalar =
            std::make_shared<IterArg>(argument->name_hint_, argument->GetType(), initial, argument->span_);
        var_remap_[argument.get()] = scalar;
        lowered->iter_args_.push_back(std::move(scalar));
        lowered->return_vars_.push_back(result);
      }
    }
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      lowered->condition_ = VisitExpr(loop->condition_);
    } else {
      lowered->attrs_ = MutateScopeAttrs(loop->attrs_).first;
    }
    YieldContext context(loop->return_vars_);
    lowered->body_ = LowerRegion(loop->body_, context);
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      if (As<TensorType>(argument->GetType())) {
        CHECK_SPAN(context.tensor_values[i] == initial_values[i], loop->span_)
            << "LowerTileToBuffer: GM loop results must retain their initial parameter alias";
        tensor_aliases_[loop->return_vars_[i].get()] = initial_values[i];
      }
      var_remap_.erase(argument.get());
    }
    return lowered;
  }

  static StmtPtr Empty(const Span& span) { return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span); }

  VarPtr Handle(const ExprPtr& value) const {
    auto tile = As<TileType>(value->GetType());
    INTERNAL_CHECK_SPAN(tile, value->span_) << "Internal error: Buffer conversion expected a Tile operand";
    const auto memory = GetDefinedMemRef(tile);
    auto found = storage_.roots.find(memory->base_.get());
    INTERNAL_CHECK_SPAN(found != storage_.roots.end(), value->span_)
        << "Internal error: missing indexed allocation for a Tile operand";
    return found->second.handle;
  }

  MakeTuplePtr Valid(const ExprPtr& tile) const {
    const auto type = As<BufferType>(Handle(tile)->GetType());
    std::vector<ExprPtr> extents;
    for (const auto extent : type->valid_shape_) {
      extents.push_back(std::make_shared<ConstInt>(extent, DataType::INDEX, tile->span_));
    }
    return std::make_shared<MakeTuple>(std::move(extents), tile->span_);
  }

  StmtPtr Operation(const std::string& name, std::vector<ExprPtr> args, const Span& span) const {
    return std::make_shared<EvalStmt>(OpRegistry::GetInstance().CreateInternal(name, args, span), span);
  }

  StmtPtr LowerCall(const CallPtr& call, const VarPtr& result) {
    INTERNAL_CHECK_SPAN(call->op_, call->span_) << "Internal error: device call has no operator";
    CHECK_SPAN(call->attrs_.empty(), call->span_)
        << "LowerTileToBuffer: device-call attributes require an explicit conversion contract";
    if (IsOp(call, "tile.alloc")) {
      INTERNAL_CHECK_SPAN(result, call->span_) << "Internal error: allocation has no pointer definition";
      auto found = storage_.roots.find(result.get());
      if (found == storage_.roots.end()) return Empty(call->span_);
      std::vector<ExprPtr> args{std::make_shared<MakeTuple>(std::vector<ExprPtr>{}, call->span_)};
      if (addressed_) args.push_back(found->second.memory->byte_offset_);
      const auto& handle = found->second.handle;
      auto allocation =
          OpRegistry::GetInstance().CreateInternal("buffer.alloc", args, {}, handle->GetType(), call->span_);
      return std::make_shared<AssignStmt>(handle, allocation, call->span_);
    }
    if (IsOp(call, "tile.create")) {
      INTERNAL_CHECK_SPAN(result, call->span_) << "Internal error: tile.create has no result";
      (void)Handle(result);
      return Empty(call->span_);
    }
    if (IsOp(call, "tile.load")) {
      INTERNAL_CHECK_SPAN(result && call->args_.size() >= 3, call->span_)
          << "Internal error: malformed tile.load";
      CHECK_SPAN(GetIntKwarg(call->kwargs_, "cache", 0) == 0, call->span_)
          << "LowerTileToBuffer: cache-policy loads require a Buffer transfer recipe";
      return Operation("buffer.load",
                       {VisitExpr(call->args_[0]), VisitExpr(call->args_[1]), Valid(result), Handle(result)},
                       call->span_);
    }
    if (IsOp(call, "tile.store")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
          << "Internal error: dense rank-2 tile.store requires three operands";
      CHECK_SPAN(
          GetIntKwarg(call->kwargs_, "atomic", 0) == 0 && GetIntKwarg(call->kwargs_, "st_phase", 0) == 0,
          call->span_)
          << "LowerTileToBuffer: atomic and phased stores require a Buffer transfer recipe";
      auto output = VisitExpr(call->args_[2]);
      if (result) tensor_aliases_[result.get()] = output;
      return Operation("buffer.store",
                       {Handle(call->args_[0]), VisitExpr(call->args_[1]), Valid(call->args_[0]), output},
                       call->span_);
    }
    if (const auto* recipe = As<GlobalVar>(call->op_)
                                 ? nullptr
                                 : backend::FindLogicalBufferElementwiseRecipe(call->op_->name_)) {
      INTERNAL_CHECK_SPAN(result && call->args_.size() == recipe->input_count, call->span_)
          << "Internal error: malformed Tile elementwise recipe operands or result";
      std::vector<ExprPtr> args;
      for (const auto& input : call->args_) args.push_back(Handle(input));
      args.push_back(Handle(result));
      auto lowered =
          OpRegistry::GetInstance().CreateInternal(recipe->buffer_op, args, call->kwargs_, call->span_);
      return std::make_shared<EvalStmt>(lowered, call->span_);
    }
    if (IsOp(call, "tile.move")) {
      INTERNAL_CHECK_SPAN(result && !call->args_.empty(), call->span_)
          << "Internal error: tile.move requires a source and a result";
      auto source = Handle(call->args_[0]);
      auto target = Handle(result);
      return source == target ? Empty(call->span_) : Operation("buffer.copy", {source, target}, call->span_);
    }
    CHECK_SPAN(false, call->span_) << "LowerTileToBuffer: no conversion recipe for '" << call->op_->name_
                                   << "'";
    return Empty(call->span_);
  }

  const StorageIndex& storage_;
  bool addressed_;
  YieldContext* yield_context_ = nullptr;
  // Input SSA uses distinct identities for branch-local definitions. Retaining
  // their mappings is linear and avoids copying the outer map at each region.
  std::unordered_map<const Var*, ExprPtr> tensor_aliases_;
};

ProgramPtr TransformProgram(const ProgramPtr& program) {
  const auto* context = PassContext::Current();
  const bool addressed = !context || context->GetMemoryPlanner() != MemoryPlanner::PtoAS;
  auto& verifiers = PropertyVerifierRegistry::GetInstance();
  verifiers.VerifyOrThrow({IRProperty::TileStorageLegalized}, program);
  if (addressed) verifiers.VerifyOrThrow({IRProperty::TileStorageAllocated}, program);
  std::vector<FunctionPtr> functions;
  functions.reserve(program->functions_.size());
  for (const auto& [global, function] : program->functions_) {
    if (!IsInCoreType(function->func_type_) || function->ir_stage_ == FunctionIRStage::Buffer) {
      functions.push_back(function);
      continue;
    }
    for (const auto& param : function->params_) {
      CHECK_SPAN(!As<TileType>(param->GetType()), param->span_)
          << "LowerTileToBuffer: device Tile parameters require the Buffer helper ABI recipe";
    }
    StorageIndex storage(addressed);
    storage.VisitStmt(function->body_);
    TileToBufferMutator mutator(storage, addressed);
    auto lowered = std::make_shared<Function>(*function);
    lowered->body_ = mutator.VisitStmt(function->body_);
    lowered->ir_stage_ = FunctionIRStage::Buffer;
    functions.push_back(std::move(lowered));
  }
  auto result = std::make_shared<Program>(functions, program->name_, program->span_);
  verifiers.VerifyOrThrow({IRProperty::BufferIR}, result);
  return result;
}

}  // namespace

namespace pass {

Pass LowerTileToBuffer() {
  return CreateProgramPass(TransformProgram, "LowerTileToBuffer", kLowerTileToBufferProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
