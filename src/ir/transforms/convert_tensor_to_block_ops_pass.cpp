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
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/op_conversion_registry.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Unwrap a single StmtPtr into a flat vector of statements.
 *
 * If the statement is a SeqStmts, returns its children; otherwise returns a single-element vector.
 */
std::vector<StmtPtr> FlattenToStmts(const StmtPtr& stmt) {
  if (auto seq = As<SeqStmts>(stmt)) {
    return seq->stmts_;
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
 * @brief Update body_map for a loop iter_arg to shadow any outer scope mapping.
 *
 * If the iter_arg's type changed, maps the name to a new Var with the updated type
 * (for substitution). Otherwise, erases any outer mapping to prevent it from leaking
 * into the loop body.
 */
void ShadowIterArgInBodyMap(std::unordered_map<std::string, VarPtr>& body_map,
                            const IterArgPtr& orig_iter_arg, const IterArgPtr& new_iter_arg) {
  if (new_iter_arg->GetType() != orig_iter_arg->GetType()) {
    body_map[orig_iter_arg->name_] =
        std::make_shared<Var>(new_iter_arg->name_, new_iter_arg->GetType(), new_iter_arg->span_);
  } else {
    body_map.erase(orig_iter_arg->name_);
  }
}

/**
 * @brief Build a MakeTuple of zeros for load/store offsets (INT64).
 */
ExprPtr MakeZeroOffsets(size_t ndim, const Span& span) {
  std::vector<ExprPtr> zeros;
  zeros.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    zeros.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
  }
  return std::make_shared<MakeTuple>(zeros, span);
}

/**
 * @brief Build a MakeTuple from a shape vector.
 */
ExprPtr MakeShapeTuple(const std::vector<ExprPtr>& shape, const Span& span) {
  return std::make_shared<MakeTuple>(shape, span);
}

/**
 * @brief Substitute variables in an expression using a name-based map.
 *
 * Recursively traverses Call, MakeTuple, BinaryExpr, UnaryExpr, and
 * TupleGetItemExpr to replace Var references.
 */
ExprPtr SubstituteExpr(const ExprPtr& expr, const std::unordered_map<std::string, VarPtr>& var_map) {
  // Check IterArg first (inherits Var but has different ObjectKind)
  if (auto iter_arg = As<IterArg>(expr)) {
    auto it = var_map.find(iter_arg->name_);
    if (it != var_map.end()) {
      return it->second;
    }
    return expr;
  }
  if (auto var = As<Var>(expr)) {
    auto it = var_map.find(var->name_);
    if (it != var_map.end()) {
      return it->second;
    }
    return expr;
  }
  if (auto call = As<Call>(expr)) {
    std::vector<ExprPtr> new_args;
    new_args.reserve(call->args_.size());
    bool changed = false;
    for (const auto& arg : call->args_) {
      auto new_arg = SubstituteExpr(arg, var_map);
      new_args.push_back(new_arg);
      if (new_arg != arg) {
        changed = true;
      }
    }
    if (!changed) {
      return expr;
    }
    return std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);
  }
  if (auto make_tuple = As<MakeTuple>(expr)) {
    std::vector<ExprPtr> new_elements;
    new_elements.reserve(make_tuple->elements_.size());
    bool changed = false;
    for (const auto& elem : make_tuple->elements_) {
      auto new_elem = SubstituteExpr(elem, var_map);
      new_elements.push_back(new_elem);
      if (new_elem != elem) {
        changed = true;
      }
    }
    if (!changed) {
      return expr;
    }
    return std::make_shared<MakeTuple>(new_elements, make_tuple->span_);
  }
  if (auto tgi = As<TupleGetItemExpr>(expr)) {
    auto new_tuple = SubstituteExpr(tgi->tuple_, var_map);
    if (new_tuple == tgi->tuple_) {
      return expr;
    }
    return std::make_shared<TupleGetItemExpr>(new_tuple, tgi->index_, tgi->span_);
  }
  // BinaryExpr/UnaryExpr are abstract with many concrete subclasses (Add, Sub, etc.),
  // so generic reconstruction is not practical. Recurse into operands to verify no
  // substitution is needed. These are scalar arithmetic expressions whose operands
  // are scalar vars/constants, not tensor/tile vars, so substitution won't fire.
  if (auto bin = As<BinaryExpr>(expr)) {
    auto new_left = SubstituteExpr(bin->left_, var_map);
    auto new_right = SubstituteExpr(bin->right_, var_map);
    INTERNAL_CHECK(new_left == bin->left_ && new_right == bin->right_)
        << "Internal error: BinaryExpr operand substitution not supported — "
        << "scalar expressions should not reference tensor/tile variables";
    return expr;
  }
  if (auto un = As<UnaryExpr>(expr)) {
    auto new_operand = SubstituteExpr(un->operand_, var_map);
    INTERNAL_CHECK(new_operand == un->operand_)
        << "Internal error: UnaryExpr operand substitution not supported — "
        << "scalar expressions should not reference tensor/tile variables";
    return expr;
  }
  // For leaf expression types (ConstInt, ConstFloat, etc.), return as-is
  return expr;
}

/**
 * @brief Find the YieldStmt in a list of statements and return its value types.
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
 * @brief Recursively transform statements in an InCore function body.
 *
 * Converts tensor ops to block ops, handling nested control flow (IfStmt, ForStmt,
 * WhileStmt, ScopeStmt).
 */
std::vector<StmtPtr> TransformIncoreBody(const std::vector<StmtPtr>& stmts,
                                         std::unordered_map<std::string, VarPtr>& tensor_to_tile,
                                         const OpConversionRegistry& conv_registry,
                                         const OpRegistry& op_registry, const Span& span) {
  std::vector<StmtPtr> result;

  for (const auto& stmt : stmts) {
    // ReturnStmt: pass through (handled by Phase 3 in TransformIncoreFunction)
    if (As<ReturnStmt>(stmt)) {
      result.push_back(stmt);
      continue;
    }

    // YieldStmt: substitute variables
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(yield->value_.size());
      bool yield_changed = false;
      for (const auto& val : yield->value_) {
        auto new_val = SubstituteExpr(val, tensor_to_tile);
        new_values.push_back(new_val);
        if (new_val != val) yield_changed = true;
      }
      if (yield_changed) {
        result.push_back(std::make_shared<YieldStmt>(new_values, yield->span_));
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // SeqStmts: recurse into children
    if (auto seq = As<SeqStmts>(stmt)) {
      auto inner = TransformIncoreBody(seq->stmts_, tensor_to_tile, conv_registry, op_registry, span);
      result.insert(result.end(), inner.begin(), inner.end());
      continue;
    }

    // ScopeStmt: recurse into body (transparent scope, defs leak through)
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto inner = TransformIncoreBody(body_stmts, tensor_to_tile, conv_registry, op_registry, span);
      result.push_back(std::make_shared<ScopeStmt>(scope->scope_kind_,
                                                   WrapInSeqStmts(inner, scope->body_->span_), scope->span_));
      continue;
    }

    // IfStmt: recurse into branches
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_condition = SubstituteExpr(if_stmt->condition_, tensor_to_tile);

      // Recurse into then branch with a copy of the map
      auto then_map = tensor_to_tile;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then_stmts = TransformIncoreBody(then_stmts, then_map, conv_registry, op_registry, span);
      auto new_then_body = WrapInSeqStmts(new_then_stmts, if_stmt->then_body_->span_);

      // Recurse into else branch with a copy of the map
      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_map = tensor_to_tile;
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else_stmts = TransformIncoreBody(else_stmts, else_map, conv_registry, op_registry, span);
        new_else_body = WrapInSeqStmts(new_else_stmts, (*if_stmt->else_body_)->span_);
      }

      // Update return_vars types based on yield types (check then branch, fall back to else)
      auto yield_types = FindYieldTypes(new_then_stmts);
      if (yield_types.empty() && new_else_body.has_value()) {
        yield_types = FindYieldTypes(FlattenToStmts(*new_else_body));
      }
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(if_stmt->return_vars_.size());
      for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
        const auto& rv = if_stmt->return_vars_[i];
        if (i < yield_types.size() && yield_types[i] != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_, yield_types[i], rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv->name_] = new_rv;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<IfStmt>(new_condition, new_then_body, new_else_body, new_return_vars,
                                                if_stmt->span_));
      continue;
    }

    // ForStmt: recurse into body
    if (auto for_stmt = As<ForStmt>(stmt)) {
      auto new_start = SubstituteExpr(for_stmt->start_, tensor_to_tile);
      auto new_stop = SubstituteExpr(for_stmt->stop_, tensor_to_tile);
      auto new_step = SubstituteExpr(for_stmt->step_, tensor_to_tile);

      // Process iter_args: substitute initValue_, update types if changed
      auto body_map = tensor_to_tile;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(for_stmt->iter_args_.size());
      for (const auto& iter_arg : for_stmt->iter_args_) {
        auto new_init = SubstituteExpr(iter_arg->initValue_, tensor_to_tile);
        auto new_ia = iter_arg;
        if (new_init->GetType() != iter_arg->GetType()) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, iter_arg->GetType(), new_init, iter_arg->span_);
        }
        new_iter_args.push_back(new_ia);
        ShadowIterArgInBodyMap(body_map, iter_arg, new_ia);
      }

      // Recurse into body
      auto body_stmts = FlattenToStmts(for_stmt->body_);
      auto new_body_stmts = TransformIncoreBody(body_stmts, body_map, conv_registry, op_registry, span);
      auto new_body = WrapInSeqStmts(new_body_stmts, for_stmt->body_->span_);

      // Update return_vars types to match iter_arg types
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
        const auto& rv = for_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv->name_] = new_rv;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<ForStmt>(for_stmt->loop_var_, new_start, new_stop, new_step,
                                                 new_iter_args, new_body, new_return_vars, for_stmt->span_,
                                                 for_stmt->kind_));
      continue;
    }

    // WhileStmt: recurse into body
    if (auto while_stmt = As<WhileStmt>(stmt)) {
      // Process iter_args: substitute initValue_, update types if changed
      auto body_map = tensor_to_tile;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(while_stmt->iter_args_.size());
      for (const auto& iter_arg : while_stmt->iter_args_) {
        auto new_init = SubstituteExpr(iter_arg->initValue_, tensor_to_tile);
        auto new_ia = iter_arg;
        if (new_init->GetType() != iter_arg->GetType()) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, iter_arg->GetType(), new_init, iter_arg->span_);
        }
        new_iter_args.push_back(new_ia);
        ShadowIterArgInBodyMap(body_map, iter_arg, new_ia);
      }

      // Substitute condition using body_map (condition references iter_arg values)
      auto new_condition = SubstituteExpr(while_stmt->condition_, body_map);

      // Recurse into body
      auto body_stmts = FlattenToStmts(while_stmt->body_);
      auto new_body_stmts = TransformIncoreBody(body_stmts, body_map, conv_registry, op_registry, span);
      auto new_body = WrapInSeqStmts(new_body_stmts, while_stmt->body_->span_);

      // Update return_vars types to match iter_arg types
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
        const auto& rv = while_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv->name_] = new_rv;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<WhileStmt>(new_condition, new_iter_args, new_body, new_return_vars,
                                                   while_stmt->span_));
      continue;
    }

    // AssignStmt: convert tensor ops to block ops
    auto assign = As<AssignStmt>(stmt);
    if (!assign) {
      // Non-assign, non-control-flow statements pass through
      result.push_back(stmt);
      continue;
    }

    auto call = As<Call>(assign->value_);
    if (!call) {
      auto new_value = SubstituteExpr(assign->value_, tensor_to_tile);
      if (new_value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        tensor_to_tile[assign->var_->name_] = new_var;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // Skip function calls (GlobalVar) — only process op calls
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (global_var) {
      auto new_value = SubstituteExpr(assign->value_, tensor_to_tile);
      if (new_value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        tensor_to_tile[assign->var_->name_] = new_var;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    const auto* converter = conv_registry.Lookup(call->op_->name_);
    if (!converter) {
      auto new_value = SubstituteExpr(assign->value_, tensor_to_tile);
      if (new_value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        tensor_to_tile[assign->var_->name_] = new_var;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // Substitute args and call the converter
    std::vector<ExprPtr> substituted_args;
    substituted_args.reserve(call->args_.size());
    for (const auto& arg : call->args_) {
      substituted_args.push_back(SubstituteExpr(arg, tensor_to_tile));
    }

    auto conv_result = (*converter)(substituted_args, call->kwargs_, call->span_);

    for (const auto& prologue_stmt : conv_result.prologue) {
      result.push_back(prologue_stmt);
    }

    std::string tile_name = assign->var_->name_ + "_tile";
    auto tile_var = std::make_shared<Var>(tile_name, conv_result.result->GetType(), assign->var_->span_);
    result.push_back(std::make_shared<AssignStmt>(tile_var, conv_result.result, assign->span_));
    tensor_to_tile[assign->var_->name_] = tile_var;
  }

  return result;
}

/**
 * @brief Transform an InCore function: insert loads, convert ops, insert stores
 *
 * @param func The InCore function to transform
 * @return Transformed function with tile ops, plus the number of added output params
 */
struct IncoreTransformResult {
  FunctionPtr func;
  size_t num_added_outputs;
};

IncoreTransformResult TransformIncoreFunction(const FunctionPtr& func) {
  auto& conv_registry = OpConversionRegistry::GetInstance();
  auto& op_registry = OpRegistry::GetInstance();
  const auto& span = func->span_;

  // Map from tensor var name -> tile var for substitution
  std::unordered_map<std::string, VarPtr> tensor_to_tile;

  // New body statements
  std::vector<StmtPtr> new_stmts;

  // Phase 1: Insert block.load for each TensorType parameter
  for (const auto& param : func->params_) {
    auto tensor_type = As<TensorType>(param->GetType());
    if (!tensor_type) {
      continue;  // ScalarType params pass through unchanged
    }

    // Create block.load(param, zeros, shape, target_memory=Vec)
    auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), span);
    auto shapes = MakeShapeTuple(tensor_type->shape_, span);
    std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec}};
    auto load_call = op_registry.Create("block.load", {param, offsets, shapes}, load_kwargs, span);

    // Create tile variable
    std::string tile_name = param->name_ + "_tile";
    auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), span);

    new_stmts.push_back(std::make_shared<AssignStmt>(tile_var, load_call, span));
    tensor_to_tile[param->name_] = tile_var;
  }

  // Phase 2: Walk body and convert tensor ops to block ops (recursive for nested control flow)
  auto body_stmts = FlattenToStmts(func->body_);

  // Separate return statement from body (will be replaced in Phase 3)
  ReturnStmtPtr return_stmt;
  std::vector<StmtPtr> non_return_stmts;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      return_stmt = ret;
    } else {
      non_return_stmts.push_back(stmt);
    }
  }

  auto transformed = TransformIncoreBody(non_return_stmts, tensor_to_tile, conv_registry, op_registry, span);
  new_stmts.insert(new_stmts.end(), transformed.begin(), transformed.end());

  // Phase 3: Add output params + block.store for return values
  INTERNAL_CHECK(return_stmt) << "Internal error: InCore function has no return statement";

  std::vector<VarPtr> new_params = func->params_;
  std::vector<TypePtr> new_return_types;
  std::vector<ExprPtr> new_return_exprs;
  size_t num_added_outputs = 0;

  for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
    auto ret_expr = SubstituteExpr(return_stmt->value_[i], tensor_to_tile);

    // Check if the return value is a tile (was converted from tensor)
    auto tile_type = As<TileType>(ret_expr->GetType());
    if (tile_type) {
      // Find the original tensor type from the function's return types
      auto orig_tensor_type = As<TensorType>(func->return_types_[i]);
      INTERNAL_CHECK(orig_tensor_type)
          << "Internal error: return type " << i << " should be TensorType but got "
          << func->return_types_[i]->TypeName();

      // Add output tensor parameter
      std::string out_name = "out_" + std::to_string(num_added_outputs);
      auto out_param = std::make_shared<Var>(out_name, orig_tensor_type, span);
      new_params.push_back(out_param);

      // Insert block.store(tile, zeros, shape, out_param)
      auto offsets = MakeZeroOffsets(tile_type->shape_.size(), span);
      auto shapes = MakeShapeTuple(tile_type->shape_, span);
      auto store_call = op_registry.Create("block.store", {ret_expr, offsets, shapes, out_param}, span);

      auto store_var = std::make_shared<Var>(out_name, store_call->GetType(), span);
      new_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, span));

      new_return_types.push_back(store_call->GetType());
      new_return_exprs.push_back(store_var);
      ++num_added_outputs;
    } else {
      // Non-tile return values pass through
      new_return_types.push_back(ret_expr->GetType());
      new_return_exprs.push_back(ret_expr);
    }
  }

  // Build new return statement
  new_stmts.push_back(std::make_shared<ReturnStmt>(new_return_exprs, return_stmt->span_));

  auto new_body = std::make_shared<SeqStmts>(new_stmts, span);
  auto new_func = std::make_shared<Function>(func->name_, new_params, new_return_types, new_body, span,
                                             FunctionType::InCore);

  return {new_func, num_added_outputs};
}

/**
 * @brief Recursively update call sites in statement lists.
 *
 * For each call to a transformed InCore function, inserts tensor.create for output params
 * and adds them as extra arguments. Handles nested control flow.
 */
std::vector<StmtPtr> UpdateCallSitesBody(
    const std::vector<StmtPtr>& stmts, std::unordered_map<std::string, VarPtr>& var_map,
    const std::unordered_map<std::string, size_t>& incore_added_outputs,
    const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs,
    const OpRegistry& op_registry, const Span& span, bool& changed) {
  std::vector<StmtPtr> result;

  for (const auto& stmt : stmts) {
    // ReturnStmt: substitute vars
    if (auto ret = As<ReturnStmt>(stmt)) {
      if (!var_map.empty()) {
        std::vector<ExprPtr> new_ret_exprs;
        new_ret_exprs.reserve(ret->value_.size());
        for (const auto& expr : ret->value_) {
          new_ret_exprs.push_back(SubstituteExpr(expr, var_map));
        }
        result.push_back(std::make_shared<ReturnStmt>(new_ret_exprs, ret->span_));
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // YieldStmt: substitute vars
    if (auto yield = As<YieldStmt>(stmt)) {
      if (!var_map.empty()) {
        std::vector<ExprPtr> new_values;
        new_values.reserve(yield->value_.size());
        for (const auto& val : yield->value_) {
          new_values.push_back(SubstituteExpr(val, var_map));
        }
        result.push_back(std::make_shared<YieldStmt>(new_values, yield->span_));
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // SeqStmts: recurse
    if (auto seq = As<SeqStmts>(stmt)) {
      auto inner = UpdateCallSitesBody(seq->stmts_, var_map, incore_added_outputs, transformed_incore_funcs,
                                       op_registry, span, changed);
      result.insert(result.end(), inner.begin(), inner.end());
      continue;
    }

    // ScopeStmt: recurse
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto inner = UpdateCallSitesBody(body_stmts, var_map, incore_added_outputs, transformed_incore_funcs,
                                       op_registry, span, changed);
      result.push_back(std::make_shared<ScopeStmt>(scope->scope_kind_,
                                                   WrapInSeqStmts(inner, scope->body_->span_), scope->span_));
      continue;
    }

    // IfStmt: recurse into branches
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_condition = SubstituteExpr(if_stmt->condition_, var_map);

      auto then_map = var_map;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then_stmts = UpdateCallSitesBody(then_stmts, then_map, incore_added_outputs,
                                                transformed_incore_funcs, op_registry, span, changed);
      auto new_then_body = WrapInSeqStmts(new_then_stmts, if_stmt->then_body_->span_);

      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_map = var_map;
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else_stmts = UpdateCallSitesBody(else_stmts, else_map, incore_added_outputs,
                                                  transformed_incore_funcs, op_registry, span, changed);
        new_else_body = WrapInSeqStmts(new_else_stmts, (*if_stmt->else_body_)->span_);
      }

      // Update return_vars types based on yield types (check then branch, fall back to else)
      auto yield_types = FindYieldTypes(new_then_stmts);
      if (yield_types.empty() && new_else_body.has_value()) {
        yield_types = FindYieldTypes(FlattenToStmts(*new_else_body));
      }
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(if_stmt->return_vars_.size());
      for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
        const auto& rv = if_stmt->return_vars_[i];
        if (i < yield_types.size() && yield_types[i] != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_, yield_types[i], rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv->name_] = new_rv;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<IfStmt>(new_condition, new_then_body, new_else_body, new_return_vars,
                                                if_stmt->span_));
      continue;
    }

    // ForStmt: recurse into body
    if (auto for_stmt = As<ForStmt>(stmt)) {
      auto new_start = SubstituteExpr(for_stmt->start_, var_map);
      auto new_stop = SubstituteExpr(for_stmt->stop_, var_map);
      auto new_step = SubstituteExpr(for_stmt->step_, var_map);

      auto body_map = var_map;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(for_stmt->iter_args_.size());
      for (const auto& iter_arg : for_stmt->iter_args_) {
        auto new_init = SubstituteExpr(iter_arg->initValue_, var_map);
        auto new_ia = iter_arg;
        if (new_init->GetType() != iter_arg->GetType()) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, iter_arg->GetType(), new_init, iter_arg->span_);
        }
        new_iter_args.push_back(new_ia);
        ShadowIterArgInBodyMap(body_map, iter_arg, new_ia);
      }

      auto body_stmts = FlattenToStmts(for_stmt->body_);
      auto new_body_stmts = UpdateCallSitesBody(body_stmts, body_map, incore_added_outputs,
                                                transformed_incore_funcs, op_registry, span, changed);
      auto new_body = WrapInSeqStmts(new_body_stmts, for_stmt->body_->span_);

      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
        const auto& rv = for_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv->name_] = new_rv;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<ForStmt>(for_stmt->loop_var_, new_start, new_stop, new_step,
                                                 new_iter_args, new_body, new_return_vars, for_stmt->span_,
                                                 for_stmt->kind_));
      continue;
    }

    // WhileStmt: recurse into body
    if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto body_map = var_map;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(while_stmt->iter_args_.size());
      for (const auto& iter_arg : while_stmt->iter_args_) {
        auto new_init = SubstituteExpr(iter_arg->initValue_, var_map);
        auto new_ia = iter_arg;
        if (new_init->GetType() != iter_arg->GetType()) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia = std::make_shared<IterArg>(iter_arg->name_, iter_arg->GetType(), new_init, iter_arg->span_);
        }
        new_iter_args.push_back(new_ia);
        ShadowIterArgInBodyMap(body_map, iter_arg, new_ia);
      }

      auto new_condition = SubstituteExpr(while_stmt->condition_, body_map);

      auto body_stmts = FlattenToStmts(while_stmt->body_);
      auto new_body_stmts = UpdateCallSitesBody(body_stmts, body_map, incore_added_outputs,
                                                transformed_incore_funcs, op_registry, span, changed);
      auto new_body = WrapInSeqStmts(new_body_stmts, while_stmt->body_->span_);

      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
        const auto& rv = while_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv->name_] = new_rv;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<WhileStmt>(new_condition, new_iter_args, new_body, new_return_vars,
                                                   while_stmt->span_));
      continue;
    }

    // AssignStmt: existing call-site update logic
    auto assign = As<AssignStmt>(stmt);
    if (!assign) {
      result.push_back(stmt);
      continue;
    }

    auto value = var_map.empty() ? assign->value_ : SubstituteExpr(assign->value_, var_map);

    auto call = As<Call>(value);
    if (!call) {
      if (value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_->name_] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!global_var) {
      if (value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_->name_] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    auto it = incore_added_outputs.find(global_var->name_);
    if (it == incore_added_outputs.end() || it->second == 0) {
      if (value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_->name_] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // This call targets a transformed InCore function — need to add output tensor args
    size_t num_outputs = it->second;
    auto incore_func_it = transformed_incore_funcs.find(global_var->name_);
    INTERNAL_CHECK(incore_func_it != transformed_incore_funcs.end())
        << "Internal error: transformed InCore function not found: " << global_var->name_;
    const auto& incore_func = incore_func_it->second;

    std::vector<ExprPtr> extra_args;
    size_t orig_param_count = incore_func->params_.size() - num_outputs;

    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& out_param = incore_func->params_[orig_param_count + i];
      auto out_tensor_type = As<TensorType>(out_param->GetType());
      INTERNAL_CHECK(out_tensor_type) << "Internal error: output param is not TensorType";

      auto shape_tuple = MakeShapeTuple(out_tensor_type->shape_, span);
      std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", out_tensor_type->dtype_}};
      auto create_call = op_registry.Create("tensor.create", {shape_tuple}, create_kwargs, span);

      std::string out_name = "out_" + std::to_string(i);
      auto out_var = std::make_shared<Var>(out_name, create_call->GetType(), span);
      result.push_back(std::make_shared<AssignStmt>(out_var, create_call, span));
      extra_args.push_back(out_var);
    }

    std::vector<ExprPtr> new_args = call->args_;
    new_args.insert(new_args.end(), extra_args.begin(), extra_args.end());

    TypePtr new_return_type;
    if (incore_func->return_types_.empty()) {
      new_return_type = nullptr;
    } else if (incore_func->return_types_.size() == 1) {
      new_return_type = incore_func->return_types_[0];
    } else {
      new_return_type = std::make_shared<TupleType>(incore_func->return_types_);
    }

    std::shared_ptr<Call> new_call;
    if (new_return_type) {
      new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, new_return_type, call->span_);
    } else {
      new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->span_);
    }

    auto new_assign_var = std::make_shared<Var>(assign->var_->name_, new_return_type, assign->var_->span_);
    result.push_back(std::make_shared<AssignStmt>(new_assign_var, new_call, assign->span_));
    var_map[assign->var_->name_] = new_assign_var;
    changed = true;
  }

  return result;
}

/**
 * @brief Update call sites in orchestration/opaque functions
 *
 * For each call to a transformed InCore function, insert tensor.create for output params
 * and add them as extra arguments. Handles nested control flow recursively.
 */
FunctionPtr UpdateCallSites(const FunctionPtr& func,
                            const std::unordered_map<std::string, size_t>& incore_added_outputs,
                            const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs) {
  auto& op_registry = OpRegistry::GetInstance();
  const auto& span = func->span_;

  auto body_stmts = FlattenToStmts(func->body_);
  bool changed = false;
  std::unordered_map<std::string, VarPtr> var_map;

  auto new_stmts = UpdateCallSitesBody(body_stmts, var_map, incore_added_outputs, transformed_incore_funcs,
                                       op_registry, span, changed);

  if (!changed) {
    return func;
  }

  auto new_body = std::make_shared<SeqStmts>(new_stmts, span);
  return std::make_shared<Function>(func->name_, func->params_, func->return_types_, new_body, span,
                                    func->func_type_);
}

}  // namespace

namespace pass {

Pass ConvertTensorToBlockOps() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 1: Transform InCore functions
    std::unordered_map<std::string, size_t> incore_added_outputs;
    std::unordered_map<std::string, FunctionPtr> transformed_incore_funcs;
    std::vector<FunctionPtr> functions_phase1;

    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        auto result = TransformIncoreFunction(func);
        incore_added_outputs[func->name_] = result.num_added_outputs;
        transformed_incore_funcs[func->name_] = result.func;
        functions_phase1.push_back(result.func);
      } else {
        functions_phase1.push_back(func);
      }
    }

    // Phase 2: Update call sites in non-InCore functions
    std::vector<FunctionPtr> functions_phase2;
    for (const auto& func : functions_phase1) {
      if (func->func_type_ != FunctionType::InCore) {
        functions_phase2.push_back(UpdateCallSites(func, incore_added_outputs, transformed_incore_funcs));
      } else {
        functions_phase2.push_back(func);
      }
    }

    return std::make_shared<Program>(functions_phase2, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ConvertTensorToBlockOps", kConvertTensorToBlockOpsProperties);
}

}  // namespace pass

// ============================================================================
// IncoreBlockOps property verifier
// ============================================================================

namespace {

/**
 * @brief Checks that InCore functions have no TensorType ops (only block/tile ops).
 */
class IncoreBlockOpsVerifier : public IRVisitor {
 public:
  explicit IncoreBlockOpsVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckTensorOp(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckTensorOp(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckTensorOp(const std::shared_ptr<const Call>& call, const Span& span) {
    // Op calls use plain Op (not GlobalVar); GlobalVar is for function calls
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (global_var) return;

    // Use op category from OpRegistry instead of brittle string prefix check
    auto& op_registry = OpRegistry::GetInstance();
    if (!op_registry.IsRegistered(call->op_->name_)) return;

    const auto& entry = op_registry.GetEntry(call->op_->name_);
    if (entry.GetOpCategory() == "TensorOp" &&
        OpConversionRegistry::GetInstance().HasConversion(call->op_->name_)) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "IncoreBlockOps", 0,
          "Tensor op '" + call->op_->name_ + "' found in InCore function (should have been converted)", span);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class IncoreBlockOpsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "IncoreBlockOps"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::InCore) continue;
      IncoreBlockOpsVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateIncoreBlockOpsPropertyVerifier() {
  return std::make_shared<IncoreBlockOpsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
