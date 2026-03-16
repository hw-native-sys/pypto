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
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// Helpers
// ============================================================================

/**
 * @brief Unwrap a StmtPtr into a flat vector of statements.
 */
std::vector<StmtPtr> FlattenToStmts(const StmtPtr& stmt) {
  if (auto seq = As<SeqStmts>(stmt)) {
    return seq->stmts_;
  }
  if (auto op_stmts = As<OpStmts>(stmt)) {
    return op_stmts->stmts_;
  }
  return {stmt};
}

/**
 * @brief Wrap a vector of statements into a single SeqStmts node.
 */
StmtPtr WrapInSeqStmts(const std::vector<StmtPtr>& stmts, const Span& span) {
  return std::make_shared<SeqStmts>(stmts, span);
}

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

/**
 * @brief Substitute variables in an expression using a name-based map.
 */
ExprPtr SubstituteExpr(const ExprPtr& expr, const std::unordered_map<std::string, VarPtr>& var_map) {
  // IterArg inherits from Var, so As<Var> handles both
  if (auto var = As<Var>(expr)) {
    auto it = var_map.find(var->name_hint_);
    return (it != var_map.end()) ? it->second : expr;
  }
  if (auto call = As<Call>(expr)) {
    std::vector<ExprPtr> new_args;
    new_args.reserve(call->args_.size());
    bool changed = false;
    for (const auto& arg : call->args_) {
      auto new_arg = SubstituteExpr(arg, var_map);
      new_args.push_back(new_arg);
      if (new_arg != arg) changed = true;
    }
    if (!changed) return expr;
    return std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);
  }
  if (auto mt = As<MakeTuple>(expr)) {
    std::vector<ExprPtr> new_elems;
    new_elems.reserve(mt->elements_.size());
    bool changed = false;
    for (const auto& e : mt->elements_) {
      auto ne = SubstituteExpr(e, var_map);
      new_elems.push_back(ne);
      if (ne != e) changed = true;
    }
    if (!changed) return expr;
    return std::make_shared<MakeTuple>(new_elems, mt->span_);
  }
  if (auto tgi = As<TupleGetItemExpr>(expr)) {
    auto new_tuple = SubstituteExpr(tgi->tuple_, var_map);
    if (new_tuple == tgi->tuple_) return expr;
    return std::make_shared<TupleGetItemExpr>(new_tuple, tgi->index_, tgi->span_);
  }
  return expr;
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

/**
 * @brief Track original ND shapes for reshape-back before tile.store.
 */
struct FlattenContext {
  std::unordered_map<std::string, VarPtr> var_map;                  // old var name -> new 2D var
  std::unordered_map<std::string, std::vector<ExprPtr>> nd_shapes;  // var name -> original ND shape
};

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
        new_values.push_back(SubstituteExpr(v, ctx.var_map));
      }
      result.push_back(std::make_shared<ReturnStmt>(new_values, ret->span_));
      continue;
    }

    // YieldStmt: substitute variables
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(yield->value_.size());
      for (const auto& v : yield->value_) {
        new_values.push_back(SubstituteExpr(v, ctx.var_map));
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

    // OpStmts: recurse
    if (auto op_stmts = As<OpStmts>(stmt)) {
      auto inner = TransformBody(op_stmts->stmts_, ctx, op_registry, span);
      result.insert(result.end(), inner.begin(), inner.end());
      continue;
    }

    // ScopeStmt: recurse into body
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto inner = TransformBody(body_stmts, ctx, op_registry, span);
      result.push_back(std::make_shared<ScopeStmt>(scope->scope_kind_,
                                                   WrapInSeqStmts(inner, scope->body_->span_), scope->span_));
      continue;
    }

    // IfStmt: recurse into branches, substitute return_vars
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_cond = SubstituteExpr(if_stmt->condition_, ctx.var_map);

      auto then_ctx = ctx;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then = TransformBody(then_stmts, then_ctx, op_registry, span);
      auto new_then_body = WrapInSeqStmts(new_then, if_stmt->then_body_->span_);

      FlattenContext else_ctx = ctx;
      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else = TransformBody(else_stmts, else_ctx, op_registry, span);
        new_else_body = WrapInSeqStmts(new_else, (*if_stmt->else_body_)->span_);
      }

      // Substitute return_vars using the branch contexts
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(if_stmt->return_vars_.size());
      for (const auto& rv : if_stmt->return_vars_) {
        auto it = then_ctx.var_map.find(rv->name_hint_);
        if (it != then_ctx.var_map.end()) {
          new_return_vars.push_back(it->second);
          ctx.var_map[rv->name_hint_] = it->second;
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
      auto new_start = SubstituteExpr(for_stmt->start_, ctx.var_map);
      auto new_stop = SubstituteExpr(for_stmt->stop_, ctx.var_map);
      auto new_step = SubstituteExpr(for_stmt->step_, ctx.var_map);

      auto body_ctx = ctx;
      // Process iter_args
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(for_stmt->iter_args_.size());
      for (const auto& ia : for_stmt->iter_args_) {
        auto new_init = SubstituteExpr(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
        }
        new_iter_args.push_back(new_ia);
        // Shadow in body_ctx to avoid outer mapping leaking
        body_ctx.var_map.erase(ia->name_hint_);
      }

      auto body_stmts = FlattenToStmts(for_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = WrapInSeqStmts(new_body_stmts, for_stmt->body_->span_);

      // Substitute return_vars and propagate to outer context
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (const auto& rv : for_stmt->return_vars_) {
        auto it = body_ctx.var_map.find(rv->name_hint_);
        if (it != body_ctx.var_map.end()) {
          new_return_vars.push_back(it->second);
          ctx.var_map[rv->name_hint_] = it->second;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<ForStmt>(for_stmt->loop_var_, new_start, new_stop, new_step,
                                                 new_iter_args, new_body, new_return_vars, for_stmt->span_,
                                                 for_stmt->kind_, for_stmt->chunk_size_,
                                                 for_stmt->chunk_policy_, for_stmt->loop_origin_));
      continue;
    }

    // WhileStmt: recurse into body, substitute return_vars
    if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto body_ctx = ctx;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(while_stmt->iter_args_.size());
      for (const auto& ia : while_stmt->iter_args_) {
        auto new_init = SubstituteExpr(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
        }
        new_iter_args.push_back(new_ia);
        body_ctx.var_map.erase(ia->name_hint_);
      }

      auto new_cond = SubstituteExpr(while_stmt->condition_, body_ctx.var_map);
      auto body_stmts = FlattenToStmts(while_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = WrapInSeqStmts(new_body_stmts, while_stmt->body_->span_);

      // Substitute return_vars and propagate to outer context
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (const auto& rv : while_stmt->return_vars_) {
        auto it = body_ctx.var_map.find(rv->name_hint_);
        if (it != body_ctx.var_map.end()) {
          new_return_vars.push_back(it->second);
          ctx.var_map[rv->name_hint_] = it->second;
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
      auto new_expr = SubstituteExpr(eval->expr_, ctx.var_map);
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
      auto new_value = SubstituteExpr(assign->value_, ctx.var_map);
      if (new_value != assign->value_) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        ctx.var_map[assign->var_->name_hint_] = new_var;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    const auto& op_name = call->op_->name_;

    // ---- tile.load on >2D tile: keep load, insert reshape ----
    // TODO(pypto): Fuse tile.load + tile.reshape into a single tile.load that
    //       directly produces the 2D tile, instead of emitting two separate
    //       instructions.  This requires extending tile.load to accept a
    //       target shape that differs from the tensor shape.
    if (op_name == "tile.load") {
      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, "tile.load result");

        // Keep the original load as-is (interfaces with ND tensor)
        std::string nd_name = assign->var_->name_hint_ + "_nd";
        auto nd_var = std::make_shared<Var>(nd_name, call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(nd_var, call, assign->span_));

        // Record original ND shape
        ctx.nd_shapes[assign->var_->name_hint_] = result_tile->shape_;

        // Insert tile.reshape(nd_var, (merged, last))
        auto shape_tuple = MakeShapeTupleFromInts({merged, last}, span);
        auto reshape_call = op_registry.Create("tile.reshape", {nd_var, shape_tuple}, span);

        auto flat_var =
            std::make_shared<Var>(assign->var_->name_hint_, reshape_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, reshape_call, assign->span_));
        ctx.var_map[assign->var_->name_hint_] = flat_var;
        continue;
      }
      // ≤2D tile.load: pass through
      result.push_back(stmt);
      continue;
    }

    // ---- tile.store: reshape back to ND before storing ----
    // TODO(pypto): Fuse tile.reshape + tile.store into a single tile.store that
    //       accepts a 2D tile and writes it directly to the ND tensor,
    //       instead of emitting a reshape followed by a store.
    if (op_name == "tile.store") {
      // tile.store args: (tile, offsets, tensor)
      if (call->args_.size() >= 3) {
        auto subst_tile = SubstituteExpr(call->args_[0], ctx.var_map);
        auto tile_type = As<TileType>(subst_tile->GetType());

        // Determine the ND shape for reshape-back:
        // 1. Prefer tracked ND shape from ctx.nd_shapes (set by tile.load/tile.create,
        //    propagated through shape-preserving ops)
        // 2. Fall back to output tensor shape (covers reduce ops and other cases
        //    where the shape was not propagated)
        const std::vector<ExprPtr>* nd_shape_ptr = nullptr;
        std::string orig_tile_name;
        if (auto var = As<Var>(call->args_[0])) {
          orig_tile_name = var->name_hint_;
          auto nd_it = ctx.nd_shapes.find(orig_tile_name);
          if (nd_it != ctx.nd_shapes.end()) {
            nd_shape_ptr = &nd_it->second;
          }
        }

        // Fall back to tensor shape if no tracked ND shape
        auto out_tensor_type = As<TensorType>(call->args_[2]->GetType());
        if (!nd_shape_ptr && out_tensor_type) {
          nd_shape_ptr = &out_tensor_type->shape_;
        }

        if (nd_shape_ptr && nd_shape_ptr->size() > 2 && tile_type && tile_type->shape_.size() <= 2) {
          const auto& orig_shape = *nd_shape_ptr;

          // Build ND shape values
          std::vector<int64_t> nd_dims;
          nd_dims.reserve(orig_shape.size());
          for (const auto& dim_expr : orig_shape) {
            nd_dims.push_back(GetStaticDim(dim_expr, "tile.store reshape-back"));
          }

          // Insert tile.reshape to restore ND shape
          auto nd_shape_tuple = MakeShapeTupleFromInts(nd_dims, span);
          auto reshape_back = op_registry.Create("tile.reshape", {subst_tile, nd_shape_tuple}, span);

          std::string nd_name = (orig_tile_name.empty() ? assign->var_->name_hint_ : orig_tile_name) + "_nd";
          auto nd_var = std::make_shared<Var>(nd_name, reshape_back->GetType(), assign->var_->span_);
          result.push_back(std::make_shared<AssignStmt>(nd_var, reshape_back, assign->span_));

          // Rebuild tile.store with the ND tile
          std::vector<ExprPtr> new_store_args;
          new_store_args.push_back(nd_var);
          for (size_t i = 1; i < call->args_.size(); ++i) {
            new_store_args.push_back(SubstituteExpr(call->args_[i], ctx.var_map));
          }
          auto new_store = op_registry.Create("tile.store", new_store_args, call->kwargs_, span);
          auto store_var =
              std::make_shared<Var>(assign->var_->name_hint_, new_store->GetType(), assign->var_->span_);
          result.push_back(std::make_shared<AssignStmt>(store_var, new_store, assign->span_));
          ctx.var_map[assign->var_->name_hint_] = store_var;
          continue;
        }
      }

      // No reshape needed — just substitute and pass through
      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      for (const auto& arg : call->args_) {
        new_args.push_back(SubstituteExpr(arg, ctx.var_map));
      }
      auto new_call = op_registry.Create("tile.store", new_args, call->kwargs_, span);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.var_map[assign->var_->name_hint_] = new_var;
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
          new_args.push_back(SubstituteExpr(call->args_[i], ctx.var_map));
        }

        auto new_call = op_registry.Create(op_name, new_args, call->kwargs_, span);
        auto flat_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, new_call, assign->span_));
        ctx.var_map[assign->var_->name_hint_] = flat_var;

        // Record original ND shape for potential store
        ctx.nd_shapes[assign->var_->name_hint_] = result_tile->shape_;
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
            new_args.push_back(SubstituteExpr(arg, ctx.var_map));
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
          ctx.var_map[assign->var_->name_hint_] = new_var;
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
        auto new_arg = SubstituteExpr(arg, ctx.var_map);
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
        ctx.var_map[assign->var_->name_hint_] = new_var;

        // Propagate ND shape for shape-preserving ops (so tile.store can reshape-back).
        // Only propagate when the output shape matches the substituted input shape
        // (element-wise ops). Reduce ops change shape and must NOT propagate.
        // Note: we compare against new_args[0] (the substituted/flattened type), not
        // input_var->GetType() (the original pre-flatten type).
        if (!new_args.empty()) {
          if (auto input_var = As<Var>(call->args_[0])) {
            auto shape_it = ctx.nd_shapes.find(input_var->name_hint_);
            if (shape_it != ctx.nd_shapes.end()) {
              auto out_tile = As<TileType>(new_call->GetType());
              auto in_tile = As<TileType>(new_args[0]->GetType());
              if (out_tile && in_tile && out_tile->shape_.size() == in_tile->shape_.size()) {
                ctx.nd_shapes[assign->var_->name_hint_] = shape_it->second;
              }
            }
          }
        }
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
  auto new_body = std::make_shared<SeqStmts>(new_stmts, span);

  // return_types_ are unchanged: InCore functions return tensors (not tiles),
  // and this pass only flattens tile ops. Tensor types are never modified.
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, span, func->func_type_);
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

    // Skip ops that are expected to work with ND tiles (load/store interface with ND tensors)
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
