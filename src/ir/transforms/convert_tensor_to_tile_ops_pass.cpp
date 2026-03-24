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
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;
using transform_utils::SubstituteExpr;

namespace {

std::string MakeTileValueName(const std::string& source_name) {
  return auto_name::BuildName(auto_name::GetBaseName(source_name), "", "tile");
}

std::string MakeOutParamName(size_t index) {
  return auto_name::BuildName("ret" + std::to_string(index), "", "out");
}

std::string MakeStoreResultName(size_t index) {
  return auto_name::BuildName("ret" + std::to_string(index), "", "store");
}

/**
 * @brief Update body_map for a loop iter_arg to shadow any outer scope mapping.
 *
 * If the iter_arg's type changed, maps the old pointer to the new iter_arg
 * (for substitution). Otherwise, erases any stale mapping for the old pointer.
 */
void ShadowIterArgInBodyMap(std::unordered_map<const Var*, VarPtr>& body_map, const IterArgPtr& orig_iter_arg,
                            const IterArgPtr& new_iter_arg) {
  if (new_iter_arg != orig_iter_arg) {
    body_map[orig_iter_arg.get()] = new_iter_arg;
  } else {
    body_map.erase(orig_iter_arg.get());
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

  [[nodiscard]] const std::unordered_set<const Var*>& GetUsed() const { return used_; }

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
        if (auto iter_arg = As<IterArg>(arg)) {
          if (As<TensorType>(iter_arg->GetType())) used_.insert(iter_arg.get());
        } else if (auto var = As<Var>(arg)) {
          if (As<TensorType>(var->GetType())) used_.insert(var.get());
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const OpConversionRegistry& conv_registry_;
  std::unordered_set<const Var*> used_;
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
std::unordered_map<const Var*, MatmulSliceInfo> PreScanSliceMatmulPatterns(
    const std::vector<StmtPtr>& stmts) {
  std::unordered_set<const Var*> slice_results;
  for (const auto& stmt : stmts) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign) continue;
    auto call = As<Call>(assign->value_);
    if (!call) continue;
    if (call->op_->name_ == "tensor.slice") {
      slice_results.insert(assign->var_.get());
    }
  }
  if (slice_results.empty()) return {};

  std::unordered_map<const Var*, MatmulSliceInfo> result;

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

    if (auto lhs_var = As<Var>(call->args_[0])) {
      if (slice_results.count(lhs_var.get())) {
        result[lhs_var.get()] = MatmulSliceInfo{false, a_trans};
      }
    }

    if (auto rhs_var = As<Var>(call->args_[1])) {
      if (slice_results.count(rhs_var.get())) {
        result[rhs_var.get()] = MatmulSliceInfo{true, b_trans};
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
                                         std::unordered_map<const Var*, VarPtr>& tensor_to_tile,
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

    // ScopeStmt: recurse into body (transparent scope, defs leak through)
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto inner = TransformIncoreBody(body_stmts, tensor_to_tile, conv_registry, op_registry, span);
      result.push_back(std::make_shared<ScopeStmt>(
          scope->scope_kind_, std::make_shared<SeqStmts>(inner, scope->body_->span_), scope->span_));
      continue;
    }

    // IfStmt: recurse into branches
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_condition = SubstituteExpr(if_stmt->condition_, tensor_to_tile);

      // Recurse into then branch with a copy of the map
      auto then_map = tensor_to_tile;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then_stmts = TransformIncoreBody(then_stmts, then_map, conv_registry, op_registry, span);
      auto new_then_body = std::make_shared<SeqStmts>(new_then_stmts, if_stmt->then_body_->span_);

      // Recurse into else branch with a copy of the map
      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_map = tensor_to_tile;
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else_stmts = TransformIncoreBody(else_stmts, else_map, conv_registry, op_registry, span);
        new_else_body = std::make_shared<SeqStmts>(new_else_stmts, (*if_stmt->else_body_)->span_);
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
          tensor_to_tile[rv.get()] = new_rv;
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
      auto new_body = std::make_shared<SeqStmts>(new_body_stmts, for_stmt->body_->span_);

      // Update return_vars types to match iter_arg types
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
        const auto& rv = for_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv.get()] = new_rv;
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
      auto new_body = std::make_shared<SeqStmts>(new_body_stmts, while_stmt->body_->span_);

      // Update return_vars types to match iter_arg types
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
        const auto& rv = while_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          tensor_to_tile[rv.get()] = new_rv;
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
        tensor_to_tile[assign->var_.get()] = new_var;
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
        tensor_to_tile[assign->var_.get()] = new_var;
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
        tensor_to_tile[assign->var_.get()] = new_var;
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
    if (call->op_->name_ == "tensor.slice" && matmul_slice_targets.count(assign->var_.get())) {
      const auto& info = matmul_slice_targets.at(assign->var_.get());
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

        std::string tile_name = MakeTileValueName(assign->var_->name_hint_);
        auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(tile_var, load_call, assign->span_));
        tensor_to_tile[assign->var_.get()] = tile_var;
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

    std::string tile_name = MakeTileValueName(assign->var_->name_hint_);
    auto tile_var = std::make_shared<Var>(tile_name, conv_result.result->GetType(), assign->var_->span_);
    result.push_back(std::make_shared<AssignStmt>(tile_var, conv_result.result, assign->span_));
    tensor_to_tile[assign->var_.get()] = tile_var;
  }

  return result;
}

class VarUseVisitor : public IRVisitor {
 public:
  explicit VarUseVisitor(const Var* target) : target_(target) {}

  [[nodiscard]] bool Found() const { return found_; }
  void CheckExpr(const ExprPtr& expr) { VisitExpr(expr); }
  void CheckStmt(const StmtPtr& stmt) { VisitStmt(stmt); }

 protected:
  void VisitExpr(const ExprPtr& expr) override {
    if (found_ || !expr) return;
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (found_ || !stmt) return;
    IRVisitor::VisitStmt(stmt);
  }

  void VisitVarLike_(const VarPtr& op) override {
    if (op.get() == target_) {
      found_ = true;
      return;
    }
  }

  void VisitExpr_(const IterArgPtr& op) override { VisitVarLike_(op); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->value_);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->expr_);
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    if (!op) return;
    for (const auto& value : op->value_) {
      VisitExpr(value);
      if (found_) return;
    }
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    if (!op) return;
    for (const auto& value : op->value_) {
      VisitExpr(value);
      if (found_) return;
    }
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    if (!op) return;
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
      if (found_) return;
    }
  }

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (!op) return;
    VisitStmt(op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->condition_);
    if (found_) return;
    VisitStmt(op->then_body_);
    if (found_ || !op->else_body_.has_value()) return;
    VisitStmt(*op->else_body_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->start_);
    if (found_) return;
    VisitExpr(op->stop_);
    if (found_) return;
    VisitExpr(op->step_);
    if (found_) return;
    if (op->chunk_size_.has_value()) {
      VisitExpr(*op->chunk_size_);
      if (found_) return;
    }
    for (const auto& iter_arg : op->iter_args_) {
      VisitExpr(iter_arg->initValue_);
      if (found_) return;
    }
    VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->condition_);
    if (found_) return;
    for (const auto& iter_arg : op->iter_args_) {
      VisitExpr(iter_arg->initValue_);
      if (found_) return;
    }
    VisitStmt(op->body_);
  }

 private:
  const Var* target_;
  bool found_ = false;
};

bool ExprUsesVar(const ExprPtr& expr, const Var* target) {
  if (!expr || !target) return false;
  VarUseVisitor visitor(target);
  visitor.CheckExpr(expr);
  return visitor.Found();
}

bool StmtUsesVar(const StmtPtr& stmt, const Var* target) {
  if (!stmt || !target) return false;
  VarUseVisitor visitor(target);
  visitor.CheckStmt(stmt);
  return visitor.Found();
}

using ParamOrigins = std::vector<size_t>;
using AliasOriginMap = std::unordered_map<const Var*, ParamOrigins>;

struct YieldAliasInfo {
  bool has_yield = false;
  std::vector<ParamOrigins> origins;
};

void AddOrigin(ParamOrigins& origins, size_t index) {
  if (std::find(origins.begin(), origins.end(), index) == origins.end()) {
    origins.push_back(index);
  }
}

void MergeOrigins(ParamOrigins& dst, const ParamOrigins& src) {
  for (size_t index : src) {
    AddOrigin(dst, index);
  }
}

void MarkAccess(const ParamOrigins& origins, std::vector<bool>& flags) {
  for (size_t index : origins) {
    if (index < flags.size()) {
      flags[index] = true;
    }
  }
}

ParamOrigins LookupOrigins(const Var* var, const AliasOriginMap& origin_map) {
  if (!var) return {};
  auto it = origin_map.find(var);
  if (it == origin_map.end()) return {};
  return it->second;
}

ParamOrigins CollectReferencedOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map);

ExprPtr GetCallKwargExpr(const CallPtr& call, const std::string& key) {
  if (!call || !call->HasKwarg(key)) return nullptr;
  return call->GetKwarg<ExprPtr>(key, ExprPtr{});
}

ExprPtr GetWriteTargetExpr(const CallPtr& call) {
  if (!call) return nullptr;

  const std::string& op_name = call->op_->name_;
  if (op_name == "tensor.write" && !call->args_.empty()) {
    return call->args_[0];
  }
  if (op_name == "tile.store") {
    if (call->args_.size() >= 3) {
      return call->args_[2];
    }
    return GetCallKwargExpr(call, "output_tensor");
  }
  if (op_name == "tensor.assemble" && !call->args_.empty()) {
    return call->args_[0];
  }
  return nullptr;
}

void UpdateTensorAliasOrigin(const VarPtr& var, const ParamOrigins& origins, AliasOriginMap& origin_map) {
  if (As<TensorType>(var->GetType()) && !origins.empty()) {
    origin_map[var.get()] = origins;
  } else {
    origin_map.erase(var.get());
  }
}

ParamOrigins GetAliasOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map) {
  if (!expr) return {};

  if (auto var = AsVarLike(expr)) {
    return LookupOrigins(var.get(), origin_map);
  }

  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    if (auto tuple = As<MakeTuple>(tuple_get->tuple_)) {
      if (tuple_get->index_ >= 0 && static_cast<size_t>(tuple_get->index_) < tuple->elements_.size()) {
        return GetAliasOrigins(tuple->elements_[static_cast<size_t>(tuple_get->index_)], origin_map);
      }
    }
    return {};
  }

  auto call = As<Call>(expr);
  if (!call) return {};

  const std::string& op_name = call->op_->name_;
  if (auto write_target = GetWriteTargetExpr(call)) {
    return GetAliasOrigins(write_target, origin_map);
  }
  if (op_name == "tensor.slice" && !call->args_.empty()) {
    return GetAliasOrigins(call->args_[0], origin_map);
  }
  return {};
}

ParamOrigins CollectReferencedOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map) {
  if (!expr) return {};

  if (auto var = AsVarLike(expr)) {
    return LookupOrigins(var.get(), origin_map);
  }

  if (auto tuple = As<MakeTuple>(expr)) {
    ParamOrigins origins;
    for (const auto& element : tuple->elements_) {
      MergeOrigins(origins, CollectReferencedOrigins(element, origin_map));
    }
    return origins;
  }

  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    if (auto tuple = As<MakeTuple>(tuple_get->tuple_)) {
      if (tuple_get->index_ >= 0 && static_cast<size_t>(tuple_get->index_) < tuple->elements_.size()) {
        return CollectReferencedOrigins(tuple->elements_[static_cast<size_t>(tuple_get->index_)], origin_map);
      }
    }
    return CollectReferencedOrigins(tuple_get->tuple_, origin_map);
  }

  auto call = As<Call>(expr);
  if (!call) return {};

  ParamOrigins origins;
  for (const auto& arg : call->args_) {
    MergeOrigins(origins, CollectReferencedOrigins(arg, origin_map));
  }
  return origins;
}

void AnalyzeCallAccess(const CallPtr& call, const AliasOriginMap& origin_map, std::vector<bool>& has_read,
                       std::vector<bool>& has_write) {
  if (!call) return;

  const std::string& op_name = call->op_->name_;
  if (op_name == "tile.load" || op_name == "tensor.read") {
    if (!call->args_.empty()) {
      MarkAccess(GetAliasOrigins(call->args_[0], origin_map), has_read);
    }
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    return;
  }

  if (op_name == "tile.store") {
    if (!call->args_.empty()) {
      MarkAccess(CollectReferencedOrigins(call->args_[0], origin_map), has_read);
    }
    for (size_t i = 1; i + 1 < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (call->args_.size() < 3) {
      MarkAccess(CollectReferencedOrigins(GetCallKwargExpr(call, "offsets"), origin_map), has_read);
    }
    if (auto write_target = GetWriteTargetExpr(call)) {
      MarkAccess(GetAliasOrigins(write_target, origin_map), has_write);
    }
    return;
  }

  if (op_name == "tensor.write") {
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (auto write_target = GetWriteTargetExpr(call)) {
      MarkAccess(GetAliasOrigins(write_target, origin_map), has_write);
    }
    return;
  }

  if (op_name == "tensor.assemble") {
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (!call->args_.empty()) {
      MarkAccess(GetAliasOrigins(call->args_[0], origin_map), has_write);
    }
    return;
  }

  if (op_name == "tensor.slice" || op_name == "tensor.create") {
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    return;
  }

  for (const auto& arg : call->args_) {
    MarkAccess(CollectReferencedOrigins(arg, origin_map), has_read);
  }
}

YieldAliasInfo MergeYieldInfos(const YieldAliasInfo& lhs, const YieldAliasInfo& rhs) {
  if (!lhs.has_yield) return rhs;
  if (!rhs.has_yield) return lhs;

  YieldAliasInfo merged;
  merged.has_yield = true;
  size_t count = std::max(lhs.origins.size(), rhs.origins.size());
  merged.origins.resize(count);
  for (size_t i = 0; i < lhs.origins.size(); ++i) {
    MergeOrigins(merged.origins[i], lhs.origins[i]);
  }
  for (size_t i = 0; i < rhs.origins.size(); ++i) {
    MergeOrigins(merged.origins[i], rhs.origins[i]);
  }
  return merged;
}

YieldAliasInfo AnalyzeStmtAliases(const StmtPtr& stmt, AliasOriginMap& origin_map,
                                  std::vector<bool>& has_read, std::vector<bool>& has_write);

YieldAliasInfo AnalyzeStmtSequenceAliases(const std::vector<StmtPtr>& stmts, AliasOriginMap& origin_map,
                                          std::vector<bool>& has_read, std::vector<bool>& has_write) {
  YieldAliasInfo last_yield;
  for (const auto& stmt : stmts) {
    auto yield_info = AnalyzeStmtAliases(stmt, origin_map, has_read, has_write);
    if (yield_info.has_yield) {
      last_yield = yield_info;
    }
  }
  return last_yield;
}

YieldAliasInfo AnalyzeStmtAliases(const StmtPtr& stmt, AliasOriginMap& origin_map,
                                  std::vector<bool>& has_read, std::vector<bool>& has_write) {
  if (!stmt) return {};

  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write);
    }

    if (As<TensorType>(assign->var_->GetType())) {
      auto origins = GetAliasOrigins(assign->value_, origin_map);
      UpdateTensorAliasOrigin(assign->var_, origins, origin_map);
    }
    return {};
  }

  if (auto eval = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(eval->expr_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write);
    }
    return {};
  }

  if (auto seq = As<SeqStmts>(stmt)) {
    return AnalyzeStmtSequenceAliases(seq->stmts_, origin_map, has_read, has_write);
  }

  if (auto scope = As<ScopeStmt>(stmt)) {
    return AnalyzeStmtAliases(scope->body_, origin_map, has_read, has_write);
  }

  if (auto if_stmt = As<IfStmt>(stmt)) {
    if (auto cond_call = As<Call>(if_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write);
    }
    auto then_map = origin_map;
    auto then_yield = AnalyzeStmtAliases(if_stmt->then_body_, then_map, has_read, has_write);
    YieldAliasInfo else_yield;
    if (if_stmt->else_body_.has_value()) {
      auto else_map = origin_map;
      else_yield = AnalyzeStmtAliases(*if_stmt->else_body_, else_map, has_read, has_write);
    }
    auto merged_yield = MergeYieldInfos(then_yield, else_yield);
    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins;
      if (merged_yield.has_yield && i < merged_yield.origins.size()) {
        origins = merged_yield.origins[i];
      }
      UpdateTensorAliasOrigin(if_stmt->return_vars_[i], origins, origin_map);
    }
    return merged_yield;
  }

  if (auto for_stmt = As<ForStmt>(stmt)) {
    if (auto start_call = As<Call>(for_stmt->start_)) {
      AnalyzeCallAccess(start_call, origin_map, has_read, has_write);
    }
    if (auto stop_call = As<Call>(for_stmt->stop_)) {
      AnalyzeCallAccess(stop_call, origin_map, has_read, has_write);
    }
    if (auto step_call = As<Call>(for_stmt->step_)) {
      AnalyzeCallAccess(step_call, origin_map, has_read, has_write);
    }
    if (for_stmt->chunk_size_.has_value()) {
      if (auto chunk_call = As<Call>(*for_stmt->chunk_size_)) {
        AnalyzeCallAccess(chunk_call, origin_map, has_read, has_write);
      }
    }

    auto body_map = origin_map;
    std::vector<ParamOrigins> init_origins(for_stmt->iter_args_.size());
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      init_origins[i] = GetAliasOrigins(for_stmt->iter_args_[i]->initValue_, origin_map);
      if (!init_origins[i].empty()) {
        body_map[for_stmt->iter_args_[i].get()] = init_origins[i];
      } else {
        body_map.erase(for_stmt->iter_args_[i].get());
      }
    }

    auto yield_info = AnalyzeStmtAliases(for_stmt->body_, body_map, has_read, has_write);
    for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins;
      if (yield_info.has_yield && i < yield_info.origins.size()) {
        origins = yield_info.origins[i];
      }
      if (origins.empty() && i < init_origins.size()) {
        origins = init_origins[i];
      }
      UpdateTensorAliasOrigin(for_stmt->return_vars_[i], origins, origin_map);
    }
    return {};
  }

  if (auto while_stmt = As<WhileStmt>(stmt)) {
    if (auto cond_call = As<Call>(while_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write);
    }

    auto body_map = origin_map;
    std::vector<ParamOrigins> init_origins(while_stmt->iter_args_.size());
    for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
      init_origins[i] = GetAliasOrigins(while_stmt->iter_args_[i]->initValue_, origin_map);
      if (!init_origins[i].empty()) {
        body_map[while_stmt->iter_args_[i].get()] = init_origins[i];
      } else {
        body_map.erase(while_stmt->iter_args_[i].get());
      }
    }

    auto yield_info = AnalyzeStmtAliases(while_stmt->body_, body_map, has_read, has_write);
    for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins;
      if (yield_info.has_yield && i < yield_info.origins.size()) {
        origins = yield_info.origins[i];
      }
      if (origins.empty() && i < init_origins.size()) {
        origins = init_origins[i];
      }
      if (As<TensorType>(while_stmt->return_vars_[i]->GetType()) && !origins.empty()) {
        origin_map[while_stmt->return_vars_[i].get()] = origins;
      } else {
        origin_map.erase(while_stmt->return_vars_[i].get());
      }
    }
    return {};
  }

  if (auto yield = As<YieldStmt>(stmt)) {
    YieldAliasInfo info;
    info.has_yield = true;
    info.origins.reserve(yield->value_.size());
    for (const auto& value : yield->value_) {
      info.origins.push_back(GetAliasOrigins(value, origin_map));
    }
    return info;
  }

  return {};
}

void UpgradeWrittenTensorParamDirections(const std::vector<StmtPtr>& stmts, const std::vector<VarPtr>& params,
                                         std::vector<ParamDirection>& param_directions) {
  std::vector<bool> has_read(params.size(), false);
  std::vector<bool> has_write(params.size(), false);
  AliasOriginMap origin_map;

  for (size_t i = 0; i < params.size() && i < param_directions.size(); ++i) {
    if (!As<TensorType>(params[i]->GetType())) {
      continue;
    }
    origin_map[params[i].get()] = ParamOrigins{i};
  }

  auto analysis_map = origin_map;
  AnalyzeStmtSequenceAliases(stmts, analysis_map, has_read, has_write);

  for (size_t i = 0; i < params.size() && i < param_directions.size(); ++i) {
    if (param_directions[i] != ParamDirection::In || !has_write[i]) {
      continue;
    }
    param_directions[i] = has_read[i] ? ParamDirection::InOut : ParamDirection::Out;
  }
}

struct ReturnedAssembleLoopRewrite {
  size_t stmt_index;
  std::optional<size_t> dead_init_stmt_index;
  ForStmtPtr new_for_stmt;
  VarPtr new_return_var;
};

std::optional<ReturnedAssembleLoopRewrite> RewriteReturnedAssembleLoopToStore(
    const std::vector<StmtPtr>& stmts, const ExprPtr& ret_expr, const VarPtr& out_param,
    const TensorTypePtr& out_tensor_type, const OpRegistry& op_registry) {
  auto ret_var = As<Var>(ret_expr);
  if (!ret_var) return std::nullopt;

  for (size_t stmt_index = 0; stmt_index < stmts.size(); ++stmt_index) {
    auto for_stmt = As<ForStmt>(stmts[stmt_index]);
    if (!for_stmt || for_stmt->iter_args_.size() != 1 || for_stmt->return_vars_.size() != 1 ||
        for_stmt->return_vars_[0].get() != ret_var.get()) {
      continue;
    }

    const auto& old_iter_arg = for_stmt->iter_args_[0];
    auto body_stmts = FlattenToStmts(for_stmt->body_);

    AssignStmtPtr assemble_assign;
    YieldStmtPtr yield_stmt;
    for (const auto& body_stmt : body_stmts) {
      auto assign = As<AssignStmt>(body_stmt);
      if (assign) {
        auto call = As<Call>(assign->value_);
        bool is_target_assemble = false;
        if (call && call->op_->name_ == "tile.assemble" && call->args_.size() == 3) {
          if (auto iter = As<IterArg>(call->args_[0])) {
            is_target_assemble = iter.get() == old_iter_arg.get();
          } else if (auto var = As<Var>(call->args_[0])) {
            is_target_assemble = var.get() == old_iter_arg.get();
          }
        }
        if (is_target_assemble) {
          if (assemble_assign) return std::nullopt;
          auto assemble_call = As<Call>(assign->value_);
          CHECK(assemble_call) << "Internal error: expected tile.assemble call in assemble loop rewrite";
          if (ExprUsesVar(assemble_call->args_[1], old_iter_arg.get()) ||
              ExprUsesVar(assemble_call->args_[2], old_iter_arg.get())) {
            return std::nullopt;
          }
          assemble_assign = assign;
          continue;
        }
      }

      if (auto yield = As<YieldStmt>(body_stmt)) {
        if (yield->value_.size() != 1 || yield_stmt) return std::nullopt;
        yield_stmt = yield;
        continue;
      }

      if (StmtUsesVar(body_stmt, old_iter_arg.get())) {
        return std::nullopt;
      }
    }

    if (!assemble_assign || !yield_stmt) return std::nullopt;

    auto yielded_var = As<Var>(yield_stmt->value_[0]);
    if (!yielded_var || yielded_var.get() != assemble_assign->var_.get()) {
      return std::nullopt;
    }

    auto assemble_call = As<Call>(assemble_assign->value_);
    CHECK(assemble_call) << "Internal error: expected tile.assemble call in assemble loop rewrite";

    auto new_iter_arg =
        std::make_shared<IterArg>(old_iter_arg->name_hint_, out_tensor_type, out_param, old_iter_arg->span_);
    auto store_call = op_registry.Create(
        "tile.store", {assemble_call->args_[1], assemble_call->args_[2], new_iter_arg}, assemble_call->span_);
    auto store_var = std::make_shared<Var>(assemble_assign->var_->name_hint_, store_call->GetType(),
                                           assemble_assign->var_->span_);

    std::vector<StmtPtr> new_body_stmts;
    new_body_stmts.reserve(body_stmts.size());
    for (const auto& body_stmt : body_stmts) {
      if (body_stmt == assemble_assign) {
        new_body_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, assemble_assign->span_));
        continue;
      }
      if (body_stmt == yield_stmt) {
        new_body_stmts.push_back(
            std::make_shared<YieldStmt>(std::vector<ExprPtr>{store_var}, yield_stmt->span_));
        continue;
      }
      new_body_stmts.push_back(body_stmt);
    }

    auto new_return_var = std::make_shared<Var>(for_stmt->return_vars_[0]->name_hint_, out_tensor_type,
                                                for_stmt->return_vars_[0]->span_);
    auto new_for_stmt =
        std::make_shared<ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_,
                                  std::vector<IterArgPtr>{new_iter_arg},
                                  std::make_shared<SeqStmts>(new_body_stmts, for_stmt->body_->span_),
                                  std::vector<VarPtr>{new_return_var}, for_stmt->span_, for_stmt->kind_,
                                  for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_);

    std::optional<size_t> dead_init_stmt_index;
    if (auto init_var = As<Var>(old_iter_arg->initValue_)) {
      bool has_other_uses = false;
      for (size_t other_index = 0; other_index < stmts.size(); ++other_index) {
        const StmtPtr& stmt_to_check = other_index == stmt_index ? new_for_stmt : stmts[other_index];
        if (StmtUsesVar(stmt_to_check, init_var.get())) {
          has_other_uses = true;
          break;
        }
      }
      if (!has_other_uses) {
        for (size_t other_index = 0; other_index < stmts.size(); ++other_index) {
          auto init_assign = As<AssignStmt>(stmts[other_index]);
          if (init_assign && init_assign->var_.get() == init_var.get()) {
            dead_init_stmt_index = other_index;
            break;
          }
        }
      }
    }

    return ReturnedAssembleLoopRewrite{
        stmt_index,
        dead_init_stmt_index,
        new_for_stmt,
        new_return_var,
    };
  }

  return std::nullopt;
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

  std::unordered_map<const Var*, VarPtr> tensor_to_tile;

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
    if (params_used_by_converted_ops.find(var.get()) == params_used_by_converted_ops.end()) {
      continue;
    }

    // Create tile.load(var, zeros, shape, valid_shapes=shape, target_memory=Vec)
    auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), span);
    auto shapes = MakeShapeTuple(tensor_type->shape_, span);
    std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                 {"transpose", false}};
    auto load_call = op_registry.Create("tile.load", {var, offsets, shapes, shapes}, load_kwargs, span);

    // Create tile variable
    std::string tile_name = MakeTileValueName(var->name_hint_);
    auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), span);

    new_stmts.push_back(std::make_shared<AssignStmt>(tile_var, load_call, span));
    tensor_to_tile[var.get()] = tile_var;
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
        std::string out_name = MakeOutParamName(num_added_outputs);
        auto out_param = std::make_shared<Var>(out_name, orig_tensor_type, span);
        new_params.push_back(out_param);
        new_param_directions.push_back(ParamDirection::Out);

        if (auto loop_rewrite = RewriteReturnedAssembleLoopToStore(new_stmts, ret_expr, out_param,
                                                                   orig_tensor_type, op_registry)) {
          new_stmts[loop_rewrite->stmt_index] = loop_rewrite->new_for_stmt;
          if (loop_rewrite->dead_init_stmt_index.has_value()) {
            new_stmts.erase(new_stmts.begin() +
                            static_cast<std::ptrdiff_t>(*loop_rewrite->dead_init_stmt_index));
          }
          new_return_types.push_back(orig_tensor_type);
          new_return_exprs.push_back(loop_rewrite->new_return_var);
          ++num_added_outputs;
          continue;
        }

        // Insert tile.store(tile, zeros, out_param)
        auto offsets = MakeZeroOffsets(tile_type->shape_.size(), span);
        auto store_call = op_registry.Create("tile.store", {ret_expr, offsets, out_param}, span);

        auto store_var =
            std::make_shared<Var>(MakeStoreResultName(num_added_outputs), store_call->GetType(), span);
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

  UpgradeWrittenTensorParamDirections(new_stmts, new_params, new_param_directions);

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
    const std::vector<StmtPtr>& stmts, std::unordered_map<const Var*, VarPtr>& var_map,
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
      result.push_back(std::make_shared<ScopeStmt>(
          scope->scope_kind_, std::make_shared<SeqStmts>(inner, scope->body_->span_), scope->span_));
      continue;
    }

    // IfStmt: recurse into branches
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_condition = SubstituteExpr(if_stmt->condition_, var_map);

      auto then_map = var_map;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then_stmts = UpdateCallSitesBody(then_stmts, then_map, incore_added_outputs,
                                                transformed_incore_funcs, op_registry, span, changed);
      auto new_then_body = std::make_shared<SeqStmts>(new_then_stmts, if_stmt->then_body_->span_);

      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_map = var_map;
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else_stmts = UpdateCallSitesBody(else_stmts, else_map, incore_added_outputs,
                                                  transformed_incore_funcs, op_registry, span, changed);
        new_else_body = std::make_shared<SeqStmts>(new_else_stmts, (*if_stmt->else_body_)->span_);
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
          var_map[rv.get()] = new_rv;
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
      auto new_body = std::make_shared<SeqStmts>(new_body_stmts, for_stmt->body_->span_);

      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
        const auto& rv = for_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv.get()] = new_rv;
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
      auto new_body = std::make_shared<SeqStmts>(new_body_stmts, while_stmt->body_->span_);

      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
        const auto& rv = while_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          var_map[rv.get()] = new_rv;
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
        var_map[assign->var_.get()] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
        var_map.erase(assign->var_.get());
      }
      continue;
    }

    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!global_var) {
      if (value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_.get()] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
        var_map.erase(assign->var_.get());
      }
      continue;
    }

    auto it = incore_added_outputs.find(global_var->name_);
    if (it == incore_added_outputs.end() || it->second == 0) {
      if (value != assign->value_) {
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, value, assign->span_));
        var_map[assign->var_.get()] = new_var;
        changed = true;
      } else {
        result.push_back(stmt);
        var_map.erase(assign->var_.get());
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

      auto out_var = std::make_shared<Var>(MakeOutParamName(i), create_call->GetType(), span);
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
    var_map[assign->var_.get()] = new_assign_var;
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
  std::unordered_map<const Var*, VarPtr> var_map;

  auto new_stmts = UpdateCallSitesBody(body_stmts, var_map, incore_added_outputs, transformed_incore_funcs,
                                       op_registry, span, changed);

  if (!changed) {
    return func;
  }

  auto new_body = std::make_shared<SeqStmts>(new_stmts, span);
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, span, func->func_type_, func->level_, func->role_);
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
