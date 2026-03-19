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
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
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
 * @brief Update body_map for a loop iter_arg to shadow any outer scope mapping.
 *
 * If the iter_arg's type changed, maps the name to a new Var with the updated type
 * (for substitution). Otherwise, erases any outer mapping to prevent it from leaking
 * into the loop body.
 */
void ShadowIterArgInBodyMap(std::unordered_map<std::string, VarPtr>& body_map,
                            const IterArgPtr& orig_iter_arg, const IterArgPtr& new_iter_arg) {
  if (new_iter_arg->GetType() != orig_iter_arg->GetType()) {
    body_map[orig_iter_arg->name_hint_] =
        std::make_shared<Var>(new_iter_arg->name_hint_, new_iter_arg->GetType(), new_iter_arg->span_);
  } else {
    body_map.erase(orig_iter_arg->name_hint_);
  }
}

/**
 * @brief Visitor that collects tensor-typed variable names used directly by converted ops.
 *
 * Traverses the IR tree via IRVisitor and records the name of every Var/IterArg argument
 * whose type is TensorType and that appears in a call to an op registered in
 * OpConversionRegistry (i.e. an op that will be converted from tensor.* to tile.*).
 *
 * Used by TransformIncoreFunction to decide which tensor parameters require a synthesised
 * default Vec-space tile.load in Phase 1.  Parameters that are only referenced by
 * non-converted ops (e.g. tile.load, tile.move) already manage their own tile
 * representation and must NOT get an extra load inserted.
 *
 * Also excludes parameters used by tensor.slice and tensor.matmul since those conversions
 * create their own block.load with proper offsets/memory spaces.
 */
class TensorArgsInConvertedOpsCollector : public IRVisitor {
 public:
  explicit TensorArgsInConvertedOpsCollector(const OpConversionRegistry& conv_registry)
      : conv_registry_(conv_registry) {}

  [[nodiscard]] const std::unordered_set<std::string>& GetUsed() const { return used_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto call = As<Call>(op->value_);
    if (call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_) &&
        conv_registry_.Lookup(call->op_->name_)) {
      // Skip ops that manage their own data loading (they create block.load
      // with specific offsets/memory-spaces during conversion, so an extra
      // Phase-1 default Vec load would be redundant or wrong).
      static const std::unordered_set<std::string> kSelfLoadingOps = {
          "tensor.slice", "tensor.matmul", "tensor.assemble", "tensor.read", "tensor.write"};
      if (kSelfLoadingOps.count(call->op_->name_)) {
        IRVisitor::VisitStmt_(op);
        return;
      }
      for (const auto& arg : call->args_) {
        // Check IterArg (subtype of Var) before Var
        if (auto iter_arg = As<IterArg>(arg)) {
          if (As<TensorType>(iter_arg->GetType())) used_.insert(iter_arg->name_hint_);
        } else if (auto var = As<Var>(arg)) {
          if (As<TensorType>(var->GetType())) used_.insert(var->name_hint_);
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const OpConversionRegistry& conv_registry_;
  std::unordered_set<std::string> used_;
};

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
 * @brief Reconstruct a BinaryExpr with new operands, dispatching on ObjectKind.
 */
ExprPtr ReconstructBinaryExpr(ObjectKind kind, const ExprPtr& left, const ExprPtr& right, DataType dtype,
                              const Span& span) {
  // clang-format off
  switch (kind) {
    case ObjectKind::Add:           return std::make_shared<Add>(left, right, dtype, span);
    case ObjectKind::Sub:           return std::make_shared<Sub>(left, right, dtype, span);
    case ObjectKind::Mul:           return std::make_shared<Mul>(left, right, dtype, span);
    case ObjectKind::FloorDiv:      return std::make_shared<FloorDiv>(left, right, dtype, span);
    case ObjectKind::FloorMod:      return std::make_shared<FloorMod>(left, right, dtype, span);
    case ObjectKind::FloatDiv:      return std::make_shared<FloatDiv>(left, right, dtype, span);
    case ObjectKind::Min:           return std::make_shared<Min>(left, right, dtype, span);
    case ObjectKind::Max:           return std::make_shared<Max>(left, right, dtype, span);
    case ObjectKind::Pow:           return std::make_shared<Pow>(left, right, dtype, span);
    case ObjectKind::Eq:            return std::make_shared<Eq>(left, right, dtype, span);
    case ObjectKind::Ne:            return std::make_shared<Ne>(left, right, dtype, span);
    case ObjectKind::Lt:            return std::make_shared<Lt>(left, right, dtype, span);
    case ObjectKind::Le:            return std::make_shared<Le>(left, right, dtype, span);
    case ObjectKind::Gt:            return std::make_shared<Gt>(left, right, dtype, span);
    case ObjectKind::Ge:            return std::make_shared<Ge>(left, right, dtype, span);
    case ObjectKind::And:           return std::make_shared<And>(left, right, dtype, span);
    case ObjectKind::Or:            return std::make_shared<Or>(left, right, dtype, span);
    case ObjectKind::Xor:           return std::make_shared<Xor>(left, right, dtype, span);
    case ObjectKind::BitAnd:        return std::make_shared<BitAnd>(left, right, dtype, span);
    case ObjectKind::BitOr:         return std::make_shared<BitOr>(left, right, dtype, span);
    case ObjectKind::BitXor:        return std::make_shared<BitXor>(left, right, dtype, span);
    case ObjectKind::BitShiftLeft:  return std::make_shared<BitShiftLeft>(left, right, dtype, span);
    case ObjectKind::BitShiftRight: return std::make_shared<BitShiftRight>(left, right, dtype, span);
    default:
      throw pypto::InternalError("ReconstructBinaryExpr: unsupported ObjectKind");
  }
  // clang-format on
}

/**
 * @brief Reconstruct a UnaryExpr with a new operand, dispatching on ObjectKind.
 */
ExprPtr ReconstructUnaryExpr(ObjectKind kind, const ExprPtr& operand, DataType dtype, const Span& span) {
  switch (kind) {
    case ObjectKind::Abs:
      return std::make_shared<Abs>(operand, dtype, span);
    case ObjectKind::Neg:
      return std::make_shared<Neg>(operand, dtype, span);
    case ObjectKind::Not:
      return std::make_shared<Not>(operand, dtype, span);
    case ObjectKind::BitNot:
      return std::make_shared<BitNot>(operand, dtype, span);
    case ObjectKind::Cast:
      return std::make_shared<Cast>(operand, dtype, span);
    default:
      throw pypto::InternalError("ReconstructUnaryExpr: unsupported ObjectKind");
  }
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
    auto it = var_map.find(iter_arg->name_hint_);
    if (it != var_map.end()) {
      return it->second;
    }
    return expr;
  }
  if (auto var = As<Var>(expr)) {
    auto it = var_map.find(var->name_hint_);
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
  if (auto bin = As<BinaryExpr>(expr)) {
    auto new_left = SubstituteExpr(bin->left_, var_map);
    auto new_right = SubstituteExpr(bin->right_, var_map);
    if (new_left == bin->left_ && new_right == bin->right_) {
      return expr;
    }
    auto dtype = GetScalarDtype(expr);
    return ReconstructBinaryExpr(bin->GetKind(), new_left, new_right, dtype, bin->span_);
  }
  if (auto un = As<UnaryExpr>(expr)) {
    auto new_operand = SubstituteExpr(un->operand_, var_map);
    if (new_operand == un->operand_) {
      return expr;
    }
    auto dtype = GetScalarDtype(expr);
    return ReconstructUnaryExpr(un->GetKind(), new_operand, dtype, un->span_);
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
 * @brief Info about a tensor.slice result that feeds into a tensor.matmul operand.
 *
 * When a tensor.slice result is consumed by tensor.matmul, the slice conversion
 * should produce tile.load(Mat, transpose=...) instead of tile.load(Vec) so that
 * the matmul conversion can skip the load and directly use the Mat-space tile.
 */
struct MatmulSliceInfo {
  bool is_rhs;     ///< true if the slice result is the rhs operand of matmul
  bool transpose;  ///< transpose flag from matmul (b_trans for rhs, a_trans for lhs)
};

/**
 * @brief Pre-scan statements to find tensor.slice results consumed by tensor.matmul.
 *
 * Scans a flat list of statements to build a map from slice result variable names
 * to their matmul usage info (which side and transpose flag).
 */
std::unordered_map<std::string, MatmulSliceInfo> PreScanSliceMatmulPatterns(
    const std::vector<StmtPtr>& stmts) {
  // Collect variable names assigned from tensor.slice
  std::unordered_set<std::string> slice_results;
  for (const auto& stmt : stmts) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign) continue;
    auto call = As<Call>(assign->value_);
    if (!call) continue;
    if (call->op_->name_ == "tensor.slice") {
      slice_results.insert(assign->var_->name_hint_);
    }
  }
  if (slice_results.empty()) return {};

  std::unordered_map<std::string, MatmulSliceInfo> result;

  // Find tensor.matmul calls that consume slice results
  for (const auto& stmt : stmts) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign) continue;
    auto call = As<Call>(assign->value_);
    if (!call || call->op_->name_ != "tensor.matmul") continue;
    if (call->args_.size() < 2) continue;

    bool a_trans = false;
    bool b_trans = false;
    for (const auto& [k, v] : call->kwargs_) {
      if (k == "a_trans") a_trans = std::any_cast<bool>(v);
      if (k == "b_trans") b_trans = std::any_cast<bool>(v);
    }

    // Check lhs (args[0])
    if (auto lhs_var = As<Var>(call->args_[0])) {
      if (slice_results.count(lhs_var->name_hint_)) {
        result[lhs_var->name_hint_] = MatmulSliceInfo{false, a_trans};
      }
    }

    // Check rhs (args[1])
    if (auto rhs_var = As<Var>(call->args_[1])) {
      if (slice_results.count(rhs_var->name_hint_)) {
        result[rhs_var->name_hint_] = MatmulSliceInfo{true, b_trans};
      }
    }
  }

  return result;
}

/**
 * @brief Recursively transform statements in an InCore function body.
 *
 * Converts tensor ops to tile ops, handling nested control flow (IfStmt, ForStmt,
 * WhileStmt, ScopeStmt).
 */
std::vector<StmtPtr> TransformIncoreBody(const std::vector<StmtPtr>& stmts,
                                         std::unordered_map<std::string, VarPtr>& tensor_to_tile,
                                         const OpConversionRegistry& conv_registry,
                                         const OpRegistry& op_registry, const Span& span) {
  std::vector<StmtPtr> result;

  auto matmul_slice_targets = PreScanSliceMatmulPatterns(stmts);

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

    // OpStmts: recurse into children (same structure as SeqStmts)
    if (auto op_stmts = As<OpStmts>(stmt)) {
      auto inner = TransformIncoreBody(op_stmts->stmts_, tensor_to_tile, conv_registry, op_registry, span);
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
          auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_types[i], rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv->name_hint_] = new_rv;
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
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
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
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv->name_hint_] = new_rv;
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
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
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
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv->name_hint_] = new_rv;
        } else {
          new_return_vars.push_back(rv);
        }
      }

      result.push_back(std::make_shared<WhileStmt>(new_condition, new_iter_args, new_body, new_return_vars,
                                                   while_stmt->span_));
      continue;
    }

    // AssignStmt: convert tensor ops to tile ops
    auto assign = As<AssignStmt>(stmt);
    if (!assign) {
      // EvalStmt: apply op conversion and var substitution (same logic as AssignStmt path)
      auto eval_stmt = As<EvalStmt>(stmt);
      if (eval_stmt) {
        auto call = As<Call>(eval_stmt->expr_);
        if (call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
          const auto* converter = conv_registry.Lookup(call->op_->name_);
          if (converter) {
            std::vector<ExprPtr> substituted_args;
            substituted_args.reserve(call->args_.size());
            for (const auto& arg : call->args_) {
              substituted_args.push_back(SubstituteExpr(arg, tensor_to_tile));
            }
            auto conv_result = (*converter)(substituted_args, call->kwargs_, call->span_);
            auto transformed_prologue =
                TransformIncoreBody(conv_result.prologue, tensor_to_tile, conv_registry, op_registry, span);
            for (const auto& prologue_stmt : transformed_prologue) {
              result.push_back(prologue_stmt);
            }
            result.push_back(std::make_shared<EvalStmt>(conv_result.result, eval_stmt->span_));
            continue;
          }
        }
        // No converter (or non-call): substitute renamed vars in args
        auto new_expr = SubstituteExpr(eval_stmt->expr_, tensor_to_tile);
        result.push_back(new_expr != eval_stmt->expr_ ? std::make_shared<EvalStmt>(new_expr, eval_stmt->span_)
                                                      : stmt);
      } else {
        // Non-assign, non-EvalStmt statements pass through unchanged
        result.push_back(stmt);
      }
      continue;
    }

    auto call = As<Call>(assign->value_);
    if (!call) {
      auto new_value = SubstituteExpr(assign->value_, tensor_to_tile);
      if (new_value != assign->value_) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        tensor_to_tile[assign->var_->name_hint_] = new_var;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // Skip function calls (GlobalVar) — only process op calls
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (global_var) {
      LOG_WARN << "[TransformIncoreBody] Skipping GlobalVar call: " << call->op_->name_;
      auto new_value = SubstituteExpr(assign->value_, tensor_to_tile);
      if (new_value != assign->value_) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        tensor_to_tile[assign->var_->name_hint_] = new_var;
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    const auto* converter = conv_registry.Lookup(call->op_->name_);
    if (!converter) {
      // TensorOps must always have a registered conversion, except for ops that are
      // handled directly by backend codegen and never need conversion in InCore bodies.
      if (op_registry.IsRegistered(call->op_->name_)) {
        const auto& entry = op_registry.GetEntry(call->op_->name_);
        static const std::unordered_set<std::string> kPassthroughTensorOps = {
            "tensor.dim",  // queries gm_tensor dimensions; backend codegen handles it directly
        };
        INTERNAL_CHECK(entry.GetOpCategory() != "TensorOp" || kPassthroughTensorOps.count(call->op_->name_))
            << "TensorOp \"" << call->op_->name_ << "\" has no registered tile conversion. "
            << "Add a conversion in src/ir/transforms/op_conversion_registry.cpp.";
      }
      auto new_value = SubstituteExpr(assign->value_, tensor_to_tile);
      if (new_value != assign->value_) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        tensor_to_tile[assign->var_->name_hint_] = new_var;
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

    // Special handling: tensor.slice feeding into tensor.matmul
    // Generate tile.load(Mat, transpose=xx) instead of the default tile.load(Vec)
    if (call->op_->name_ == "tensor.slice" && matmul_slice_targets.count(assign->var_->name_hint_)) {
      const auto& info = matmul_slice_targets.at(assign->var_->name_hint_);
      const auto& input = substituted_args[0];
      auto tensor_type = As<TensorType>(input->GetType());
      if (tensor_type) {
        // Use the slice's offset and shape args (args[1]=shape, args[2]=offset)
        const auto& shape_arg = substituted_args[1];
        const auto& offset_arg = substituted_args[2];

        // For transpose, swap shape dims: [N,K] → [K,N]
        ExprPtr load_shapes = shape_arg;
        ExprPtr valid_shapes_base = (substituted_args.size() == 4) ? substituted_args[3] : shape_arg;
        if (info.transpose) {
          auto shape_tuple = As<MakeTuple>(shape_arg);
          if (shape_tuple && shape_tuple->elements_.size() == 2) {
            std::vector<ExprPtr> swapped = {shape_tuple->elements_[1], shape_tuple->elements_[0]};
            load_shapes = std::make_shared<MakeTuple>(swapped, shape_arg->span_);
          }
          auto valid_tuple = As<MakeTuple>(valid_shapes_base);
          if (valid_tuple && valid_tuple->elements_.size() == 2) {
            std::vector<ExprPtr> swapped = {valid_tuple->elements_[1], valid_tuple->elements_[0]};
            valid_shapes_base = std::make_shared<MakeTuple>(swapped, valid_shapes_base->span_);
          }
        }

        auto valid_shapes = valid_shapes_base;
        std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Mat},
                                                                     {"transpose", info.transpose}};
        auto load_call = op_registry.Create("tile.load", {input, offset_arg, load_shapes, valid_shapes},
                                            load_kwargs, span);

        std::string tile_name = assign->var_->name_hint_ + "_tile";
        auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(tile_var, load_call, assign->span_));
        tensor_to_tile[assign->var_->name_hint_] = tile_var;
        continue;
      }
    }

    auto conv_result = (*converter)(substituted_args, call->kwargs_, call->span_);

    // Prologue statements may themselves contain tensor ops (e.g. tensor.create
    // used as a scratch buffer). Run them through the same conversion pipeline.
    auto transformed_prologue =
        TransformIncoreBody(conv_result.prologue, tensor_to_tile, conv_registry, op_registry, span);
    for (const auto& prologue_stmt : transformed_prologue) {
      result.push_back(prologue_stmt);
    }

    std::string tile_name = assign->var_->name_hint_ + "_tile";
    auto tile_var = std::make_shared<Var>(tile_name, conv_result.result->GetType(), assign->var_->span_);
    result.push_back(std::make_shared<AssignStmt>(tile_var, conv_result.result, assign->span_));
    tensor_to_tile[assign->var_->name_hint_] = tile_var;
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

  // Phase 1: Insert tile.load for each TensorType parameter that is directly consumed
  // by a converted tensor op.  Parameters that are only referenced by non-converted ops
  // (e.g. tile.load, tile.move) already manage their own tile representation and must
  // NOT get an additional Vec-space load inserted here.
  TensorArgsInConvertedOpsCollector collector(conv_registry);
  collector.VisitStmt(func->body_);
  const auto& params_used_by_converted_ops = collector.GetUsed();

  for (const auto& var : func->params_) {
    auto tensor_type = As<TensorType>(var->GetType());
    if (!tensor_type) {
      continue;  // ScalarType params pass through unchanged
    }

    // Only synthesise a default Vec load when the parameter is directly passed to an op
    // that has a registered tensor-to-tile converter.  If the function body already
    // uses the parameter via explicit tile ops (e.g. tile.load to Mat space), skip it.
    if (params_used_by_converted_ops.find(var->name_hint_) == params_used_by_converted_ops.end()) {
      continue;
    }

    // Create tile.load(var, zeros, shape, valid_shapes=shape, target_memory=Vec)
    auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), span);
    auto shapes = MakeShapeTuple(tensor_type->shape_, span);
    std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                 {"transpose", false}};
    auto load_call = op_registry.Create("tile.load", {var, offsets, shapes, shapes}, load_kwargs, span);

    // Create tile variable
    std::string tile_name = var->name_hint_ + "_tile";
    auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), span);

    new_stmts.push_back(std::make_shared<AssignStmt>(tile_var, load_call, span));
    tensor_to_tile[var->name_hint_] = tile_var;
  }

  // Phase 2: Walk body and convert tensor ops to tile ops (recursive for nested control flow)
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

  // Phase 3: Add output params + tile.store for return values
  std::vector<VarPtr> new_params = func->params_;
  std::vector<ParamDirection> new_param_directions = func->param_directions_;
  std::vector<TypePtr> new_return_types;
  size_t num_added_outputs = 0;

  if (return_stmt) {
    std::vector<ExprPtr> new_return_exprs;

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
        std::string out_name = "outbuf" + std::to_string(num_added_outputs);
        auto out_param = std::make_shared<Var>(out_name, orig_tensor_type, span);
        new_params.push_back(out_param);
        new_param_directions.push_back(ParamDirection::Out);

        // Insert tile.store(tile, zeros, out_param)
        auto offsets = MakeZeroOffsets(tile_type->shape_.size(), span);
        auto store_call = op_registry.Create("tile.store", {ret_expr, offsets, out_param}, span);

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
  } else {
    // Void function (e.g. cross-core producer): add empty return
    INTERNAL_CHECK(func->return_types_.empty())
        << "Internal error: function '" << func->name_ << "' has no ReturnStmt but declares "
        << func->return_types_.size() << " return type(s) — possible malformed IR";
    new_stmts.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, span));
  }

  auto new_body = std::make_shared<SeqStmts>(new_stmts, span);
  auto new_func = std::make_shared<Function>(func->name_, new_params, new_param_directions, new_return_types,
                                             new_body, span, FunctionType::InCore);

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

    // OpStmts: recurse (same structure as SeqStmts)
    if (auto op_stmts = As<OpStmts>(stmt)) {
      auto inner = UpdateCallSitesBody(op_stmts->stmts_, var_map, incore_added_outputs,
                                       transformed_incore_funcs, op_registry, span, changed);
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
          auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_types[i], rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv->name_hint_] = new_rv;
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
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
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
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv->name_hint_] = new_rv;
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

    // WhileStmt: recurse into body
    if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto body_map = var_map;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(while_stmt->iter_args_.size());
      for (const auto& iter_arg : while_stmt->iter_args_) {
        auto new_init = SubstituteExpr(iter_arg->initValue_, var_map);
        auto new_ia = iter_arg;
        if (new_init->GetType() != iter_arg->GetType()) {
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, new_init->GetType(), new_init, iter_arg->span_);
        } else if (new_init != iter_arg->initValue_) {
          new_ia =
              std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
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
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv->name_hint_] = new_rv;
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
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_->name_hint_] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
        // Erase any stale mapping: this assignment redefines the variable
        var_map.erase(assign->var_->name_hint_);
      }
      continue;
    }

    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!global_var) {
      if (value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_->name_hint_] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
        var_map.erase(assign->var_->name_hint_);
      }
      continue;
    }

    auto it = incore_added_outputs.find(global_var->name_);
    if (it == incore_added_outputs.end() || it->second == 0) {
      if (value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_->name_hint_] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
        var_map.erase(assign->var_->name_hint_);
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
      TensorLayout layout = out_tensor_type->tensor_view_.has_value() ? out_tensor_type->tensor_view_->layout
                                                                      : TensorLayout::ND;
      std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", out_tensor_type->dtype_},
                                                                     {"layout", layout}};
      auto create_call = op_registry.Create("tensor.create", {shape_tuple}, create_kwargs, span);

      std::string out_name = "outbuf" + std::to_string(i);
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

    auto new_assign_var =
        std::make_shared<Var>(assign->var_->name_hint_, new_return_type, assign->var_->span_);
    result.push_back(std::make_shared<AssignStmt>(new_assign_var, new_call, assign->span_));
    var_map[assign->var_->name_hint_] = new_assign_var;
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
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, span, func->func_type_);
}

}  // namespace

namespace pass {

Pass ConvertTensorToTileOps() {
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

  return CreateProgramPass(pass_func, "ConvertTensorToTileOps", kConvertTensorToTileOpsProperties);
}

}  // namespace pass

// ============================================================================
// IncoreTileOps property verifier
// ============================================================================

namespace {

/**
 * @brief Checks that InCore functions have no TensorType ops (only tile ops).
 */
class IncoreTileOpsVerifier : public IRVisitor {
 public:
  explicit IncoreTileOpsVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

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
      // tensor.read/tensor.write on a gm_tensor (TensorType input) intentionally stays unconverted
      if ((call->op_->name_ == "tensor.read" || call->op_->name_ == "tensor.write") && !call->args_.empty() &&
          As<TensorType>(call->args_[0]->GetType())) {
        return;
      }

      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "IncoreTileOps", 0,
          "Tensor op '" + call->op_->name_ + "' found in InCore function (should have been converted)", span);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class IncoreTileOpsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "IncoreTileOps"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::InCore) continue;
      IncoreTileOpsVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateIncoreTileOpsPropertyVerifier() {
  return std::make_shared<IncoreTileOpsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
