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
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;
using transform_utils::Substitute;

namespace {

// ============================================================================
// Helpers
// ============================================================================

/**
 * @brief Check if a TileType has >2 dimensions.
 */
bool IsNdTile(const TileTypePtr& tile_type) { return tile_type && tile_type->shape_.size() > 2; }

/**
 * @brief Extract a static int64_t from a ConstInt expression.
 *
 * Raises CHECK if the expression is not a ConstInt (dynamic shape).
 */
int64_t GetStaticDim(const ExprPtr& expr, const std::string& context) {
  auto ci = As<ConstInt>(expr);
  CHECK(ci) << "FlattenTileNdTo2D: all tile dimensions must be static (ConstInt), "
            << "but found dynamic dimension in " << context;
  return ci->value_;
}

/**
 * @brief Compute the merged 2D shape from an ND shape.
 *
 * [A, B, C, D] -> {A*B*C, D}
 */
std::pair<int64_t, int64_t> ComputeMergedShape(const std::vector<ExprPtr>& shape,
                                               const std::string& context) {
  int64_t merged = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    int64_t dim = GetStaticDim(shape[i], context);
    CHECK(dim > 0) << "FlattenTileNdTo2D: tile dimension " << i << " must be positive in " << context
                   << ", got " << dim;
    // Overflow check: merged * dim must fit in int64_t
    CHECK(merged <= INT64_MAX / dim) << "FlattenTileNdTo2D: integer overflow when computing merged dimension "
                                     << "in " << context << " (merged=" << merged << ", dim=" << dim << ")";
    merged *= dim;
  }
  int64_t last = GetStaticDim(shape.back(), context);
  return {merged, last};
}

/**
 * @brief Build a MakeTuple from int64_t values.
 */
ExprPtr MakeShapeTupleFromInts(const std::vector<int64_t>& dims, const Span& span) {
  std::vector<ExprPtr> elems;
  elems.reserve(dims.size());
  for (auto d : dims) {
    elems.push_back(std::make_shared<ConstInt>(d, DataType::INDEX, span));
  }
  return std::make_shared<MakeTuple>(elems, span);
}

/**
 * @brief Build a 2D shape vector from merged dimensions.
 */
std::vector<ExprPtr> Make2DShapeExprs(int64_t merged, int64_t last, const Span& span) {
  return {std::make_shared<ConstInt>(merged, DataType::INDEX, span),
          std::make_shared<ConstInt>(last, DataType::INDEX, span)};
}

// ============================================================================
// Precondition validation
// ============================================================================

/**
 * @brief Visitor that validates preconditions for the FlattenTileNdTo2D pass.
 *
 * Checks:
 * 1. All tile shapes are static (ConstInt)
 * 2. All tile reduce ops (tile.sum/max/min) on >2D tiles reduce the last axis
 * 3. No tile.read/tile.write/tile.slice on >2D tiles
 */
class PreconditionChecker : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckCall(call);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckCall(call);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  static void CheckStaticShape(const TileTypePtr& tile_type, const std::string& op_name) {
    if (!tile_type || tile_type->shape_.size() <= 2) return;
    for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
      CHECK(As<ConstInt>(tile_type->shape_[i]))
          << "FlattenTileNdTo2D: tile dimension " << i << " must be static (ConstInt) "
          << "for tile op '" << op_name << "'";
    }
  }

  void CheckCall(const CallPtr& call) {
    if (!call || !call->op_) return;
    auto gv = As<GlobalVar>(call->op_);
    if (gv) return;  // Skip function calls

    const auto& name = call->op_->name_;
    if (name.substr(0, 5) != "tile.") return;

    // Check static shapes on any tile-typed argument and result
    for (const auto& arg : call->args_) {
      CheckStaticShape(As<TileType>(arg->GetType()), name);
    }
    CheckStaticShape(As<TileType>(call->GetType()), name);

    // Disallow tile.read/tile.write/tile.slice on >2D tiles
    if (name == "tile.read" || name == "tile.write" || name == "tile.slice") {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        CHECK(!IsNdTile(input_tile)) << "FlattenTileNdTo2D: " << name << " is not supported on >2D tiles";
      }
    }

    // Check reduce ops reduce the last axis
    if (name == "tile.sum" || name == "tile.max" || name == "tile.min") {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        if (IsNdTile(input_tile)) {
          int axis = call->GetKwarg<int>("axis", -1);
          int last_axis = static_cast<int>(input_tile->shape_.size()) - 1;
          CHECK(axis == last_axis) << "FlattenTileNdTo2D: tile reduce op '" << name
                                   << "' must reduce along the last axis "
                                   << "(axis=" << last_axis << "), but got axis=" << axis;
          // keepdim must be True so the output stays 2D after flatten
          bool keepdim = call->GetKwarg<bool>("keepdim", false);
          CHECK(keepdim) << "FlattenTileNdTo2D: tile reduce op '" << name
                         << "' on >2D tile must use keepdim=True to maintain 2D output shape";
        }
      }
    }
  }
};

// ============================================================================
// Main transformation
// ============================================================================

struct FlattenContext {
  std::unordered_map<const Var*, VarPtr> var_map;  // old Var* -> new 2D var

  void Insert(const VarPtr& old_var, const VarPtr& new_var) { var_map[old_var.get()] = new_var; }

  void Erase(const VarPtr& var) { var_map.erase(var.get()); }
};

/**
 * @brief Extract yield value types from the first YieldStmt found in a statement list.
 *
 * Recurses into SeqStmts and ScopeStmt to find yields in nested containers.
 */
std::vector<TypePtr> FindYieldTypes(const std::vector<StmtPtr>& stmts) {
  for (const auto& stmt : stmts) {
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<TypePtr> types;
      types.reserve(yield->value_.size());
      for (const auto& val : yield->value_) {
        types.push_back(val->GetType());
      }
      return types;
    }
    if (auto seq = As<SeqStmts>(stmt)) {
      auto found = FindYieldTypes(seq->stmts_);
      if (!found.empty()) return found;
    }
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto found = FindYieldTypes(body_stmts);
      if (!found.empty()) return found;
    }
  }
  return {};
}

/**
 * @brief Recursively transform statements, flattening >2D tile ops to 2D.
 */
std::vector<StmtPtr> TransformBody(const std::vector<StmtPtr>& stmts, FlattenContext& ctx,
                                   const OpRegistry& op_registry, const Span& span) {
  std::vector<StmtPtr> result;

  for (const auto& stmt : stmts) {
    // ReturnStmt: substitute return values
    if (auto ret = As<ReturnStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(ret->value_.size());
      for (const auto& v : ret->value_) {
        new_values.push_back(Substitute(v, ctx.var_map));
      }
      result.push_back(std::make_shared<ReturnStmt>(new_values, ret->span_));
      continue;
    }

    // YieldStmt: substitute variables
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(yield->value_.size());
      for (const auto& v : yield->value_) {
        new_values.push_back(Substitute(v, ctx.var_map));
      }
      result.push_back(std::make_shared<YieldStmt>(new_values, yield->span_));
      continue;
    }

    // SeqStmts: recurse
    if (auto seq = As<SeqStmts>(stmt)) {
      auto inner = TransformBody(seq->stmts_, ctx, op_registry, span);
      result.insert(result.end(), inner.begin(), inner.end());
      continue;
    }

    // ScopeStmt: recurse into body
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto inner = TransformBody(body_stmts, ctx, op_registry, span);
      result.push_back(std::make_shared<ScopeStmt>(scope->scope_kind_,
                                                   SeqStmts::Flatten(std::move(inner), scope->body_->span_),
                                                   scope->span_, scope->level_, scope->role_, scope->split_));
      continue;
    }

    // IfStmt: recurse into branches, substitute return_vars
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_cond = Substitute(if_stmt->condition_, ctx.var_map);

      auto then_ctx = ctx;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then = TransformBody(then_stmts, then_ctx, op_registry, span);
      // Extract yield types before moving the vector
      auto yield_types = FindYieldTypes(new_then);
      auto new_then_body = SeqStmts::Flatten(std::move(new_then), if_stmt->then_body_->span_);

      FlattenContext else_ctx = ctx;
      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else = TransformBody(else_stmts, else_ctx, op_registry, span);
        new_else_body = SeqStmts::Flatten(std::move(new_else), (*if_stmt->else_body_)->span_);
      }

      // Update return_vars types based on yield types (positional matching)
      if (yield_types.empty() && new_else_body.has_value()) {
        yield_types = FindYieldTypes(FlattenToStmts(*new_else_body));
      }
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(if_stmt->return_vars_.size());
      for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
        const auto& rv = if_stmt->return_vars_[i];
        if (i < yield_types.size() && yield_types[i] != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_types[i], rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(
          std::make_shared<IfStmt>(new_cond, new_then_body, new_else_body, new_return_vars, if_stmt->span_));
      continue;
    }

    // ForStmt: recurse into body, substitute return_vars
    if (auto for_stmt = As<ForStmt>(stmt)) {
      auto new_start = Substitute(for_stmt->start_, ctx.var_map);
      auto new_stop = Substitute(for_stmt->stop_, ctx.var_map);
      auto new_step = Substitute(for_stmt->step_, ctx.var_map);

      auto body_ctx = ctx;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(for_stmt->iter_args_.size());
      for (const auto& ia : for_stmt->iter_args_) {
        auto new_init = Substitute(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
          body_ctx.Insert(ia, new_ia);
        } else {
          body_ctx.Erase(ia);
        }
        new_iter_args.push_back(new_ia);
      }

      auto body_stmts = FlattenToStmts(for_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(new_body_stmts), for_stmt->body_->span_);

      // Update return_vars types to match iter_arg types (positional matching)
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
        const auto& rv = for_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<ForStmt>(for_stmt->loop_var_, new_start, new_stop, new_step,
                                                 new_iter_args, new_body, new_return_vars, for_stmt->span_,
                                                 for_stmt->kind_, for_stmt->chunk_config_, for_stmt->attrs_));
      continue;
    }

    // WhileStmt: recurse into body, substitute return_vars
    if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto body_ctx = ctx;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(while_stmt->iter_args_.size());
      for (const auto& ia : while_stmt->iter_args_) {
        auto new_init = Substitute(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
          body_ctx.Insert(ia, new_ia);
        } else {
          body_ctx.Erase(ia);
        }
        new_iter_args.push_back(new_ia);
      }

      auto new_cond = Substitute(while_stmt->condition_, body_ctx.var_map);
      auto body_stmts = FlattenToStmts(while_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(new_body_stmts), while_stmt->body_->span_);

      // Update return_vars types to match iter_arg types (positional matching)
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
        const auto& rv = while_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(
          std::make_shared<WhileStmt>(new_cond, new_iter_args, new_body, new_return_vars, while_stmt->span_));
      continue;
    }

    // EvalStmt: substitute variables in the expression
    if (auto eval = As<EvalStmt>(stmt)) {
      auto new_expr = Substitute(eval->expr_, ctx.var_map);
      if (new_expr != eval->expr_) {
        // Re-create tile ops via OpRegistry for proper type deduction
        if (auto call = As<Call>(new_expr)) {
          if (call->op_ && call->op_->name_.substr(0, 5) == "tile.") {
            auto new_call = op_registry.Create(call->op_->name_, call->args_, call->kwargs_, span);
            result.push_back(std::make_shared<EvalStmt>(new_call, eval->span_));
            continue;
          }
        }
        result.push_back(std::make_shared<EvalStmt>(new_expr, eval->span_));
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // AssignStmt: the main transformation logic
    auto assign = As<AssignStmt>(stmt);
    if (!assign) {
      result.push_back(stmt);
      continue;
    }

    auto call = As<Call>(assign->value_);
    auto global_var = call ? As<GlobalVar>(call->op_) : nullptr;

    // Non-call assignment or function call (GlobalVar): substitute and pass through
    if (!call || global_var) {
      auto new_value = Substitute(assign->value_, ctx.var_map);
      if (new_value != assign->value_) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        ctx.Insert(assign->var_, new_var);
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    const auto& op_name = call->op_->name_;

    // ---- tile.load on >2D tile: produce 2D tile directly ----
    // tile.load semantics: ND tensor → 2D tile. No reshape needed.
    if (op_name == "tile.load") {
      // Substitute args via ctx.var_map so all operand Vars reference the latest SSA values.
      std::vector<ExprPtr> sub_args;
      sub_args.reserve(call->args_.size());
      for (const auto& arg : call->args_) {
        sub_args.push_back(Substitute(arg, ctx.var_map));
      }

      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, "tile.load result");

        // Construct call with explicit 2D TileType (bypasses ND type inference).
        // Create a 2D tile_view and preserve memory_space for type consistency
        // with downstream ops (op_registry always adds tile_view + memory_space).
        auto flat_shape_exprs = Make2DShapeExprs(merged, last, span);
        // Assign the implicit TileView for the flattened 2D shape+memory_space.
        // This ensures print→parse roundtrip stability: the printer omits TileView
        // fields that match the implicit defaults, and C++ type inference on reparse
        // produces the same implicit TileView, so structural_equal sees identical types.
        auto flat_tile_view = std::make_optional(
            tile_view_semantics::GetImplicitTileView(flat_shape_exprs, result_tile->memory_space_));
        auto flat_tile_type = std::make_shared<TileType>(flat_shape_exprs, result_tile->dtype_, std::nullopt,
                                                         flat_tile_view, result_tile->memory_space_);
        auto flat_call =
            std::make_shared<Call>(call->op_, sub_args, call->kwargs_, flat_tile_type, call->span_);
        auto flat_var = std::make_shared<Var>(assign->var_->name_hint_, flat_tile_type, assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, flat_call, assign->span_));
        ctx.Insert(assign->var_, flat_var);
        continue;
      }
      // ≤2D tile.load: honor any pending var_map substitutions
      auto new_call = op_registry.Create(op_name, sub_args, call->kwargs_, span);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.store: pass through, injecting shapes for ND tensors ----
    // tile.store semantics: (flattened-)2D tile → ND tensor. No reshape needed.
    // When the output tensor is ND (rank > 2), inject an explicit shapes tuple as
    // args[3] so that the codegen can reconstruct the correct pto.partition_view.
    // The shapes tuple is the tile's original shape left-padded with 1s to reach
    // the tensor rank.  Examples:
    //   tile [A,B,C] (ND, rank 3) → tensor rank 3: shapes = (A,B,C)       [no pad]
    //   tile [A,B,C] (ND, rank 3) → tensor rank 4: shapes = (1,A,B,C)     [1 pad]
    //   tile [H,W]   (2D, rank 2) → tensor rank 3: shapes = (1,H,W)       [1 pad]
    // The original tile type is read BEFORE substitution so it still carries the
    // pre-flatten ND shape.
    // Signature: (tile, offsets, output_tensor[, shapes])
    if (op_name == "tile.store") {
      auto orig_tile_type = As<TileType>(call->args_[0]->GetType());

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size() + 1);
      // Push all original args (tile, offsets, output_tensor) with substitution
      for (const auto& arg : call->args_) {
        new_args.push_back(Substitute(arg, ctx.var_map));
      }
      // Inject shapes tuple whenever the output tensor is ND (rank > 2).
      // Codegen always requires args[3] in that case regardless of tile rank.
      auto out_tensor_type = As<TensorType>(new_args[2]->GetType());
      if (orig_tile_type && out_tensor_type && out_tensor_type->shape_.size() > 2) {
        const size_t tensor_rank = out_tensor_type->shape_.size();
        const size_t tile_rank = orig_tile_type->shape_.size();
        std::vector<ExprPtr> shapes;
        shapes.reserve(tensor_rank);
        // Left-pad with 1s when tile rank < tensor rank.
        for (size_t i = tile_rank; i < tensor_rank; ++i) {
          shapes.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
        }
        for (const auto& dim : orig_tile_type->shape_) {
          shapes.push_back(dim);
        }
        new_args.push_back(std::make_shared<MakeTuple>(shapes, span));
      }

      // Construct call directly: store result type = output tensor type (args[2])
      auto out_type = new_args[2]->GetType();
      auto new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, out_type, call->span_);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.create / tile.full with >2D shape: flatten shape directly ----
    if (op_name == "tile.create" || op_name == "tile.full") {
      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, op_name);

        // Rebuild the call with 2D shape
        auto new_shape_tuple = MakeShapeTupleFromInts({merged, last}, span);
        std::vector<ExprPtr> new_args;
        // First arg is the shape tuple
        new_args.push_back(new_shape_tuple);
        // Remaining args (e.g., fill value for tile.full)
        for (size_t i = 1; i < call->args_.size(); ++i) {
          new_args.push_back(Substitute(call->args_[i], ctx.var_map));
        }

        auto new_call = op_registry.Create(op_name, new_args, call->kwargs_, span);
        auto flat_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, new_call, assign->span_));
        ctx.Insert(assign->var_, flat_var);
        continue;
      }
      // ≤2D: pass through
      result.push_back(stmt);
      continue;
    }

    // ---- tile.sum/tile.max/tile.min: remap axis to 1 (last axis of 2D) ----
    if (op_name == "tile.sum" || op_name == "tile.max" || op_name == "tile.min") {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        if (IsNdTile(input_tile)) {
          // Substitute args
          std::vector<ExprPtr> new_args;
          new_args.reserve(call->args_.size());
          for (const auto& arg : call->args_) {
            new_args.push_back(Substitute(arg, ctx.var_map));
          }

          // Update axis kwarg to 1 (last axis of 2D tile)
          std::vector<std::pair<std::string, std::any>> new_kwargs;
          for (const auto& [key, val] : call->kwargs_) {
            if (key == "axis") {
              new_kwargs.emplace_back("axis", 1);
            } else {
              new_kwargs.emplace_back(key, val);
            }
          }

          auto new_call = op_registry.Create(op_name, new_args, new_kwargs, span);
          auto new_var =
              std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
          result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
          ctx.Insert(assign->var_, new_var);
          continue;
        }
      }
    }

    // ---- All other tile ops (including tile.reshape) and non-tile ops: substitute args ----
    {
      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      bool changed = false;
      for (const auto& arg : call->args_) {
        auto new_arg = Substitute(arg, ctx.var_map);
        new_args.push_back(new_arg);
        if (new_arg != arg) changed = true;
      }

      if (!changed) {
        result.push_back(stmt);
      } else {
        // Re-create tile ops via OpRegistry for proper type deduction with 2D args;
        // non-tile ops keep the original type.
        auto new_call =
            (op_name.substr(0, 5) == "tile.")
                ? op_registry.Create(op_name, new_args, call->kwargs_, span)
                : std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);

        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
        ctx.Insert(assign->var_, new_var);
      }
    }
  }

  return result;
}

/**
 * @brief Transform a single InCore function: flatten >2D tiles to 2D.
 */
FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  const auto& span = func->span_;
  auto& op_registry = OpRegistry::GetInstance();

  // Validate preconditions
  PreconditionChecker checker;
  checker.VisitStmt(func->body_);

  // Transform body
  FlattenContext ctx;
  auto body_stmts = FlattenToStmts(func->body_);
  auto new_stmts = TransformBody(body_stmts, ctx, op_registry, span);
  auto new_body = SeqStmts::Flatten(std::move(new_stmts), span);

  // return_types_ are unchanged: InCore functions return tensors (not tiles),
  // and this pass only flattens tile ops. Tensor types are never modified.
  auto result =
      std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                 new_body, span, func->func_type_, func->level_, func->role_, func->attrs_);
  return result;
}

// ============================================================================
// Property Verifier
// ============================================================================

/**
 * @brief Visitor that checks all tile ops in InCore functions use ≤2D tiles.
 */
class TileOps2DVerifier : public IRVisitor {
 public:
  explicit TileOps2DVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckCall(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckCall(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckCall(const CallPtr& call, const Span& stmt_span) {
    if (!call || !call->op_) return;
    auto gv = As<GlobalVar>(call->op_);
    if (gv) return;

    const auto& name = call->op_->name_;
    if (name.substr(0, 5) != "tile.") return;

    // tile.load/tile.store are permitted to have any tile rank:
    // load produces 2D tiles from ND tensors; store accepts 2D tiles and writes to ND tensors.
    if (name == "tile.load" || name == "tile.store" || name == "tile.reshape") return;

    // Check result type
    auto result_tile = As<TileType>(call->GetType());
    if (result_tile && result_tile->shape_.size() > 2) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                "Tile op '" + name + "' in InCore function '" + func_name_ +
                                    "' produces >2D tile (should have been flattened to 2D)",
                                stmt_span);
    }

    // Check argument types
    for (const auto& arg : call->args_) {
      auto arg_tile = As<TileType>(arg->GetType());
      if (arg_tile && arg_tile->shape_.size() > 2) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                  "Tile op '" + name + "' in InCore function '" + func_name_ +
                                      "' has >2D tile argument (should have been flattened to 2D)",
                                  stmt_span);
        break;
      }
    }
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

}  // namespace

// ============================================================================
// Property Verifier Impl (public)
// ============================================================================

class TileOps2DPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TileOps2D"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (!IsInCoreType(func->func_type_)) continue;
      TileOps2DVerifier verifier(diagnostics, func->name_);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateTileOps2DPropertyVerifier() {
  return std::make_shared<TileOps2DPropertyVerifierImpl>();
}

// ============================================================================
// Pass Factory
// ============================================================================

namespace pass {

Pass FlattenTileNdTo2D() {
  return CreateFunctionPass(TransformFunction, "FlattenTileNdTo2D", kFlattenTileNdTo2DProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
