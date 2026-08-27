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
 * @file block_nz_tensor_views_pass.cpp
 * @brief BlockNzTensorViews pass — turn a logical NZ tensor into its blocked form.
 *
 * ``pl.Tensor[[E, N, K], pl.INT8, pl.NZ]`` asserts that the bytes in GM are
 * already in PTO-native NZ fractal order while keeping the *logical* shape and
 * slicing at the DSL level. pto-isa describes such a buffer with a blocked
 * rank-(r+2) GlobalTensor (``pto/common/pto_tile.hpp``):
 *
 *     shape   = [E, K/c0, N/16, 16, c0]
 *     strides = [K*N,     N*c0, 16*c0, c0, 1]      (c0 = 256 / dtype bits)
 *
 * This pass rewrites the IR into exactly that form:
 *
 *   Phase 1 — every TensorType tagged ``TensorLayout::NZ`` gets its shape
 *             replaced by ``BlockNzShape``. The stride slot is left empty for
 *             ``MaterializeTensorStrides`` (pass 30) to fill; because a blocked
 *             NZ shape's row-major strides *are* pto-isa's NZ strides, that
 *             pass needs no NZ-specific rule.
 *
 *   Phase 2 — every ``tile.load`` reading such a tensor gets its offsets /
 *             shapes / valid_shape rewritten into blocked coordinates, while
 *             its result ``TileType`` is preserved verbatim: the GM partition
 *             becomes rank-(r+2) but the destination tile stays the logical
 *             2-D ``[N_TILE, K_TILE]``.
 *
 * After this pass no logical-shaped NZ TensorType survives, so nothing
 * downstream — including codegen, which derives every rank from
 * ``TensorType::shape_`` — needs to know NZ is special. That is the reason the
 * blocking lives here rather than in the backend: ``EmitMakeTensorViews``,
 * ``GetTensorViewTypeString`` and the ``tile.load`` ``partition_view`` emitter
 * each read the rank independently and must agree.
 *
 * Ordering constraints (see docs/en/dev/passes/14-block_nz_tensor_views.md):
 *   * after ConvertTensorToTileOps / LowerCompositeOps — the ``tile.load`` ops
 *     Phase 2 rewrites must already exist;
 *   * after FlattenTileNdTo2D — declared as a ``TileOps2D`` requirement. The
 *     destination tile must already be the logical 2-D operand: blocking a
 *     still-ND-rank tile leaves a ``tile.load`` whose type annotation and
 *     argument ranks cannot both be printed, which the printer round-trip
 *     rejects. FlattenTileNdTo2D skips its ND2NZ source-window collapse for an
 *     NZ source, so the logical window is still intact when this pass runs.
 *
 * Milestone 1 scope: read-only, matmul operands only (``target_memory=Mat``),
 * whole-byte dtypes, static shapes, fractal-aligned shapes and slice offsets.
 * Everything outside that is rejected with a diagnostic naming the authoring
 * fix — an NZ tensor must never be silently mis-addressed.
 */

#include <any>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Function-level stamp marking that Phase 1 already ran on this function.
///
/// Blocking is *not* idempotent — running it twice would block an already
/// blocked shape — and the structural ``IsBlockedNzShape`` test cannot tell a
/// blocked shape from a logical one that merely ends in ``[16, c0]``. The stamp
/// makes re-entry a no-op without relying on that ambiguity.
constexpr const char* kNzBlockedAttr = "nz_tensor_views_blocked";

/// True when a tensor-like type carries an NZ TensorView.
bool IsNzTensorType(const TypePtr& type) {
  auto tensor_type = AsTensorTypeLike(type);
  if (!tensor_type) return false;
  // Bind the optional before dereferencing so the access is locally provable
  // (bugprone-unchecked-optional-access does not follow a `&&` short-circuit).
  const auto& view = tensor_type->tensor_view_;
  return view.has_value() && view->layout == TensorLayout::NZ;
}

/// Rewrite an NZ-tagged TensorType / DistributedTensorType to its blocked
/// shape, recursing into TupleType. Returns the input unchanged when there is
/// nothing to block (identity-comparable by the caller).
TypePtr BlockNzType(const TypePtr& type, const Span& span) {
  if (!type) return type;

  if (auto tuple_type = As<TupleType>(type)) {
    std::vector<TypePtr> new_elements;
    new_elements.reserve(tuple_type->types_.size());
    bool changed = false;
    for (const auto& element : tuple_type->types_) {
      auto new_element = BlockNzType(element, span);
      if (new_element.get() != element.get()) changed = true;
      new_elements.push_back(std::move(new_element));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(new_elements));
  }

  if (auto dist_type = As<DistributedTensorType>(type)) {
    if (!IsNzTensorType(type)) return type;
    // A distributed NZ tensor would additionally need remote_load blocking,
    // which milestone 1 does not implement; remote_load.cpp rejects it. Refuse
    // here too so the diagnostic names the annotation rather than surfacing
    // later as an opaque layout error.
    CHECK_SPAN(false, span) << "NZ layout is not supported on a distributed tensor yet. "
                            << "Annotate the tensor as pl.ND or pl.DN.";
  }

  if (auto tensor_type = As<TensorType>(type)) {
    if (!IsNzTensorType(type)) return type;
    // ``IsNzTensorType`` already established the view exists, but that is not
    // locally provable, so repeat the test where the optional is dereferenced.
    const auto& maybe_view = tensor_type->tensor_view_;
    if (!maybe_view.has_value()) return type;
    auto blocked_shape = tensor_view_semantics::BlockNzShape(tensor_type->shape_, tensor_type->dtype_, span);
    // valid_shape is a per-dim companion of the logical shape; a partial NZ
    // region has no blocked representation in milestone 1.
    const TensorView& view = *maybe_view;
    CHECK_SPAN(view.valid_shape.empty(), span)
        << "NZ layout does not support a partial valid_shape yet; the whole tensor must be valid.";
    CHECK_SPAN(view.stride.empty(), span)
        << "NZ layout does not support an explicit stride yet: the blocked NZ stride is derived from "
        << "the shape. Drop the stride annotation.";
    return std::make_shared<TensorType>(std::move(blocked_shape), tensor_type->dtype_, tensor_type->memref_,
                                        view);
  }

  return type;
}

/// Rewrite the elements of a ``MakeTuple`` argument through ``fn``.
ExprPtr BlockTupleArg(const ExprPtr& arg, DataType dtype, const Span& span, bool is_offsets) {
  auto tuple = As<MakeTuple>(arg);
  INTERNAL_CHECK_SPAN(tuple, span) << "Internal error: tile.load coordinate argument must be a MakeTuple";
  auto blocked = is_offsets ? tensor_view_semantics::BlockNzOffsets(tuple->elements_, dtype, span)
                            : tensor_view_semantics::BlockNzShape(tuple->elements_, dtype, span);
  return std::make_shared<MakeTuple>(std::move(blocked), tuple->span_);
}

class BlockNzMutator : public IRMutator {
 public:
  void AddSubstitution(const VarPtr& old_var, const VarPtr& new_var) { var_cache_[old_var] = new_var; }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) return it->second;
    auto new_type = BlockNzType(op->GetType(), op->span_);
    if (new_type.get() == op->GetType().get()) {
      var_cache_[op] = op;
      return op;
    }
    auto new_var = std::make_shared<Var>(op->name_hint_, std::move(new_type), op->span_);
    var_cache_[op] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) return it->second;
    auto new_init = IRMutator::VisitExpr(op->initValue_);
    auto new_type = BlockNzType(op->GetType(), op->span_);
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

    // Scan *every* operand, not just the first. An NZ tensor can appear in any
    // position — `tile.store`'s destination is argument 2 — and phase 1 has
    // already blocked its type by the time we get here. An operand this pass
    // does not recognise must be rejected rather than left with logical
    // coordinates pointing into a blocked tensor.
    std::vector<size_t> nz_args;
    for (size_t i = 0; i < new_args.size(); ++i) {
      auto tensor = AsVarLike(new_args[i]);
      if (tensor && IsNzTensorType(tensor->GetType())) nz_args.push_back(i);
    }

    // A call to another function just forwards the tensor; the callee's own
    // params are blocked when that function is transformed.
    const bool is_function_call = static_cast<bool>(As<GlobalVar>(op->op_));
    if (!nz_args.empty() && !is_function_call) {
      // Name the store case directly: annotating an Out/InOut tensor pl.NZ is
      // the likely authoring mistake, and "read-only" is the actionable fact.
      CHECK_SPAN(!IsOp(op, "tile.store"), op->span_)
          << "NZ layout is read-only: an NZ tensor cannot be a store destination. "
          << "Annotate the output tensor as pl.ND.";
      CHECK_SPAN(IsOp(op, "tile.load") && nz_args.size() == 1 && nz_args[0] == 0, op->span_)
          << "NZ layout currently supports only 'tile.load' reading the tensor as its source, but it is "
          << "used by '" << op->op_->name_ << "' at argument " << nz_args[0]
          << ". NZ tensors are read-only matmul operands in this release.";
      args_changed = true;
      new_args = BlockTileLoadArgs(op, std::move(new_args));
    }

    auto new_return_type = BlockNzType(op->GetType(), op->span_);
    const bool type_changed = new_return_type.get() != op->GetType().get();
    if (!args_changed && !type_changed) return op;

    // Direct ctor, not OpRegistry::Create: re-deducing ``tile.load``'s type
    // from the now rank-(r+2) shapes argument would turn the destination tile
    // into a rank-(r+2) TileType. The GM partition is blocked; the tile is not.
    return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, op->attrs_,
                                  std::move(new_return_type), op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_var_expr = IRMutator::VisitExpr(op->var_);
    auto new_value = IRMutator::VisitExpr(op->value_);
    auto new_var = As<Var>(new_var_expr);
    INTERNAL_CHECK(new_var) << "Internal error: BlockNzTensorViews visited an AssignStmt LHS to a non-Var";
    if (new_var.get() == op->var_.get() && new_value.get() == op->value_.get()) return op;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

 private:
  /// Rewrite ``tile.load``'s (offsets, shapes, [valid_shape]) into blocked
  /// coordinates and enforce the milestone-1 scope guards.
  std::vector<ExprPtr> BlockTileLoadArgs(const CallPtr& op, std::vector<ExprPtr> args) {
    INTERNAL_CHECK_SPAN(args.size() >= 3, op->span_)
        << "Internal error: tile.load expects at least (tensor, offsets, shapes), got " << args.size();
    auto tensor = AsVarLike(args[0]);
    auto tensor_type = AsTensorTypeLike(tensor->GetType());
    const DataType dtype = tensor_type->dtype_;

    // pto-isa only offers NZ->NZ into a Mat tile for the matmul operand path
    // (docs/isa/TLOAD.md); an NZ source loaded into a Vec tile is a different
    // (unimplemented) lowering. ``target_memory`` is optional on tile.load, so
    // an omitted target is a rejection too — the Vec default would be wrong.
    std::optional<MemorySpace> target;
    for (const auto& [key, value] : op->kwargs_) {
      if (key == "target_memory") {
        target = AnyCast<MemorySpace>(value, "target_memory");
        break;
      }
    }
    CHECK_SPAN(target.has_value() && *target == MemorySpace::Mat, op->span_)
        << "NZ layout currently supports only matmul operand loads (target_memory=pl.Mem.Mat), got "
        << (target.has_value() ? MemorySpaceToString(*target) : std::string("no target_memory"))
        << ". An NZ tensor is a cube weight: load it into Mat, or annotate the tensor as pl.ND.";

    args[1] = BlockTupleArg(args[1], dtype, op->span_, /*is_offsets=*/true);
    args[2] = BlockTupleArg(args[2], dtype, op->span_, /*is_offsets=*/false);
    if (args.size() >= 4) {
      args[3] = BlockTupleArg(args[3], dtype, op->span_, /*is_offsets=*/false);
    }
    return args;
  }

  std::unordered_map<VarPtr, VarPtr> var_cache_;
};

/// Block one function: params, return types, body. Returns the input unchanged
/// when the function carries no NZ tensor.
FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (func->HasAttr(kNzBlockedAttr)) return func;

  bool params_changed = false;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  std::unordered_map<VarPtr, VarPtr> param_substitutions;
  for (const auto& old_param : func->params_) {
    auto new_type = BlockNzType(old_param->GetType(), old_param->span_);
    if (new_type.get() == old_param->GetType().get()) {
      new_params.push_back(old_param);
      continue;
    }
    auto new_param = std::make_shared<Var>(old_param->name_hint_, std::move(new_type), old_param->span_);
    new_params.push_back(new_param);
    param_substitutions.emplace(old_param, new_param);
    params_changed = true;
  }

  bool returns_changed = false;
  std::vector<TypePtr> new_return_types;
  new_return_types.reserve(func->return_types_.size());
  for (const auto& rt : func->return_types_) {
    auto new_rt = BlockNzType(rt, func->span_);
    if (new_rt.get() != rt.get()) returns_changed = true;
    new_return_types.push_back(std::move(new_rt));
  }

  BlockNzMutator mutator;
  for (const auto& [old_var, new_var] : param_substitutions) {
    mutator.AddSubstitution(old_var, new_var);
  }
  StmtPtr new_body = func->body_;
  if (func->body_) new_body = mutator.VisitStmt(func->body_);
  const bool body_changed = new_body.get() != func->body_.get();

  if (!params_changed && !returns_changed && !body_changed) return func;

  auto new_func = MutableCopy(func);
  if (params_changed) new_func->params_ = std::move(new_params);
  if (returns_changed) new_func->return_types_ = std::move(new_return_types);
  if (body_changed) new_func->body_ = std::move(new_body);
  new_func->attrs_.emplace_back(kNzBlockedAttr, std::any(true));
  return new_func;
}

}  // namespace

namespace pass {

Pass BlockNzTensorViews() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    bool modified = false;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func);
      if (new_func.get() != func.get()) modified = true;
      new_functions[gvar] = std::move(new_func);
    }
    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "BlockNzTensorViews", kBlockNzTensorViewsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
