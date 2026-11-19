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
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/inline_functions.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/simplify.h"
#include "pypto/ir/transforms/utils/ssa_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/unroll_loops.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/transforms/utils/vf_helpers.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/reg_scope.h"

namespace pypto::ir {
namespace {

// One index per function distinguishes region-local definitions from uses in
// its caller. Identity (including IterArg), never spelling, identifies values.
class RegionIndex : public IRVisitor {
 public:
  std::map<const Var*, std::set<const RegScopeStmt*>> uses;

 protected:
  void VisitStmt_(const RegScopeStmtPtr& scope) override {
    const auto previous = region_;
    region_ = scope.get();
    IRVisitor::VisitStmt_(scope);
    region_ = previous;
  }
  void VisitStmt_(const AssignStmtPtr& stmt) override { VisitExpr(stmt->value_); }
  void VisitExpr_(const VarPtr& var) override { uses[var.get()].insert(region_); }
  void VisitExpr_(const IterArgPtr& var) override { uses[var.get()].insert(region_); }

 private:
  const RegScopeStmt* region_ = nullptr;
};

class Operands : public var_collectors::VarDefUseCollector {
 public:
  explicit Operands(const std::set<const Var*>* available = nullptr) : available_(available) {}
  std::map<const Var*, VarPtr> objects;
  std::map<const Var*, VarPtr> aliases;
  std::vector<VarPtr> writes;
  std::set<const Var*> captured;
  std::vector<std::pair<VarPtr, VarPtr>> branch_sources;

 protected:
  void VisitExpr_(const VarPtr& var) override { Use(var); }
  void VisitExpr_(const IterArgPtr& var) override { Use(var); }
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    objects.emplace(stmt->var_.get(), stmt->var_);
    // A conditional rebinding implicitly reads the incoming value on paths
    // that skip the assignment (including a zero-trip loop).
    if (conditional_depth_ && available_ && available_->count(stmt->var_.get()) &&
        !var_defs.count(stmt->var_.get())) {
      Use(stmt->var_);
    }
    if (auto call = As<Call>(stmt->value_);
        IsOp(call, "reg.store") || IsOp(call, "reg.scatter") || IsOp(call, "reg.squeeze_store_step") ||
        IsOp(call, "reg.squeeze_store_flush")) {
      auto target = AsVarLike(GetRegMemoryEffect(call).target);
      CHECK_SPAN(target, call->span_) << "Register store requires a named Tile";
      if (As<TileType>(stmt->var_->GetType()) && stmt->var_ != target) {
        if (conditional_depth_ && aliases.count(stmt->var_.get())) {
          branch_sources.emplace_back(aliases.at(stmt->var_.get()), target);
        }
        aliases[stmt->var_.get()] = target;
      }
      writes.push_back(target);
    }
    // Source-level rebinding reads the old binding before defining its result.
    VisitExpr(stmt->value_);
    if (var_defs.insert(stmt->var_.get()).second) var_defs_ordered.push_back(stmt->var_.get());
  }

  void VisitStmt_(const ForStmtPtr& stmt) override {
    objects.emplace(stmt->loop_var_.get(), stmt->loop_var_);
    LoopAliases(stmt->iter_args_, stmt->return_vars_, stmt->body_);
    ++conditional_depth_;
    var_collectors::VarDefUseCollector::VisitStmt_(stmt);
    --conditional_depth_;
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    LoopAliases(stmt->iter_args_, stmt->return_vars_, stmt->body_);
    ++conditional_depth_;
    var_collectors::VarDefUseCollector::VisitStmt_(stmt);
    --conditional_depth_;
  }
  void VisitStmt_(const IfStmtPtr& stmt) override {
    auto lhs = transform_utils::GetLastYieldStmt(stmt->then_body_);
    auto rhs = stmt->else_body_ ? transform_utils::GetLastYieldStmt(*stmt->else_body_) : nullptr;
    for (size_t i = 0; i < stmt->return_vars_.size(); ++i) {
      const auto& result = stmt->return_vars_[i];
      objects.emplace(result.get(), result);
      if (!As<TileType>(result->GetType())) continue;
      CHECK_SPAN(lhs && rhs && i < lhs->value_.size() && i < rhs->value_.size(), stmt->span_)
          << "Register Tile branch results require yields from both arms";
      auto left = AsVarLike(lhs->value_[i]);
      auto right = AsVarLike(rhs->value_[i]);
      CHECK_SPAN(left && right, stmt->span_) << "Register Tile branch yields must be named views";
      aliases[result.get()] = left;
      branch_sources.emplace_back(left, right);
    }
    ++conditional_depth_;
    var_collectors::VarDefUseCollector::VisitStmt_(stmt);
    --conditional_depth_;
  }

 private:
  const std::set<const Var*>* available_;
  size_t conditional_depth_ = 0;
  void LoopAliases(const std::vector<IterArgPtr>& args, const std::vector<VarPtr>& results,
                   const StmtPtr& body) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    for (size_t i = 0; i < args.size(); ++i) {
      objects.emplace(args[i].get(), args[i]);
      if (As<TileType>(args[i]->GetType())) {
        auto seed = AsVarLike(args[i]->initValue_);
        CHECK_SPAN(seed, args[i]->span_) << "Register Tile loop seed must be an external view";
        aliases[args[i].get()] = seed;
        if (i < results.size()) aliases[results[i].get()] = args[i];
        CHECK_SPAN(yield && i < yield->value_.size(), args[i]->span_)
            << "Register Tile loop carries require explicit yields";
        auto next = AsVarLike(yield->value_[i]);
        CHECK_SPAN(next, args[i]->span_) << "Register Tile loop yields must be named views";
        branch_sources.emplace_back(seed, next);
      }
    }
    for (const auto& result : results) objects.emplace(result.get(), result);
  }
  void Use(const VarPtr& var) {
    objects.emplace(var.get(), var);
    if (!var_defs.count(var.get())) captured.insert(var.get());
    if (!var_uses.insert(var.get()).second) return;
    // Tile metadata is not visited by the ordinary operand visitor. Capture
    // its scalar dependencies too, especially a dynamic view origin.
    if (auto tile = As<TileType>(var->GetType())) {
      for (const auto& dim : tile->shape_) VisitExpr(dim);
      if (tile->tile_view_) {
        for (const auto& dim : tile->tile_view_->valid_shape) VisitExpr(dim);
        for (const auto& stride : tile->tile_view_->stride) VisitExpr(stride);
        if (tile->tile_view_->start_offset) VisitExpr(tile->tile_view_->start_offset);
      }
    }
    // A declaration must bind symbols before a later Tile parameter uses them.
    var_uses_ordered.push_back(var.get());
  }
};

// Normalize only the new helper: the caller remains in the early frontend
// phase. This reuses SSA conversion for temporal rebinding and branch merges.
FunctionPtr NormalizeHelper(const FunctionPtr& source) {
  // SSA would turn accumulator rebindings into iter_args, which static
  // unrolling deliberately rejects. Use the normal frontend order locally.
  auto normalized = ConvertFunctionToSSA(transform_utils::UnrollFunction(source));
  // Caller constants and unrolled indices must choose their reachable branch
  // before its Tile aliases become a shared-storage return contract. Reuse the
  // guarded folds so effects, yields and required hardware loops survive.
  normalized = transform_utils::SimplifyFunction(normalized, /*constants_only=*/true,
                                                 /*fold_constant_control_flow=*/true);
  auto sequence = As<SeqStmts>(normalized->body_);
  INTERNAL_CHECK_SPAN(sequence && sequence->stmts_.size() == 2, source->span_)
      << "Internal error: outlined helper lost its scope/return structure";
  auto scope = As<RegScopeStmt>(sequence->stmts_[0]);
  Operands operands;
  operands.VisitStmt(scope->body_);
  // Memoized store-version roots bound chained aliases to O(N log N).
  std::map<const Var*, VarPtr> roots;
  auto root = [&](VarPtr var) {
    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    while (operands.aliases.count(var.get())) {
      if (auto it = roots.find(var.get()); it != roots.end()) {
        var = it->second;
        break;
      }
      CHECK_SPAN(seen.insert(var.get()).second, scope->span_) << "Cyclic register Tile alias";
      path.push_back(var);
      var = operands.aliases.at(var.get());
    }
    for (const auto& alias : path) roots[alias.get()] = var;
    return var;
  };
  for (const auto& [left, right] : operands.branch_sources) {
    CHECK_SPAN(root(left) == root(right), scope->span_)
        << "Register Tile control-flow results require a shared external view";
  }
  std::set<const Var*> written;
  for (const auto& target : operands.writes) written.insert(root(target).get());
  auto result = MutableCopy(normalized);
  std::map<const Var*, VarPtr> params;
  for (size_t i = 0; i < result->params_.size(); ++i) {
    const auto& param = result->params_[i];
    params.emplace(param.get(), param);
    // Partial/masked stores preserve untouched contents, so writes borrow InOut storage.
    result->param_directions_[i] = written.count(param.get()) ? ParamDirection::InOut : ParamDirection::In;
  }
  std::vector<ExprPtr> returns;
  std::vector<TypePtr> types;
  for (const auto& value : As<ReturnStmt>(sequence->stmts_[1])->value_) {
    auto source_var = root(AsVarLike(value));
    CHECK_SPAN(params.count(source_var.get()), value->span_)
        << "Register Tile result must alias an external capture";
    returns.push_back(params.at(source_var.get()));
    types.push_back(returns.back()->GetType());
  }
  result->return_types_ = std::move(types);
  result->body_ = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{scope, std::make_shared<ReturnStmt>(returns, source->span_)}, source->span_);
  return result;
}

class Outliner : public IRMutator {
 public:
  Outliner(FunctionPtr function, const RegionIndex& index, std::set<std::string>& names,
           std::vector<FunctionPtr>& helpers)
      : function_(std::move(function)), index_(index), names_(names), helpers_(helpers) {
    for (const auto& param : function_->params_) available_.insert(param.get());
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& stmt) override {
    auto result = IRMutator::VisitStmt_(stmt);
    available_.insert(stmt->var_.get());
    return result;
  }
  StmtPtr VisitStmt_(const ForStmtPtr& stmt) override {
    available_.insert(stmt->loop_var_.get());
    for (const auto& arg : stmt->iter_args_) available_.insert(arg.get());
    auto result = IRMutator::VisitStmt_(stmt);
    for (const auto& var : stmt->return_vars_) available_.insert(var.get());
    return result;
  }
  StmtPtr VisitStmt_(const WhileStmtPtr& stmt) override {
    for (const auto& arg : stmt->iter_args_) available_.insert(arg.get());
    auto result = IRMutator::VisitStmt_(stmt);
    for (const auto& var : stmt->return_vars_) available_.insert(var.get());
    return result;
  }
  StmtPtr VisitStmt_(const IfStmtPtr& stmt) override {
    auto result = IRMutator::VisitStmt_(stmt);
    for (const auto& var : stmt->return_vars_) available_.insert(var.get());
    return result;
  }
  StmtPtr VisitStmt_(const RegScopeStmtPtr& scope) override {
    Operands operands(&available_);
    operands.VisitStmt(scope->body_);
    std::vector<VarPtr> captures;
    for (const auto* raw : operands.var_uses_ordered) {
      if (operands.captured.count(raw)) captures.push_back(operands.objects.at(raw));
    }
    std::unordered_map<const Var*, ExprPtr> substitutions;
    // Scalars first in the substitution table, not in the signature: a Tile
    // parameter type may refer to a scalar captured later in lexical order.
    for (const auto& capture : captures) {
      CHECK_SPAN(As<TileType>(capture->GetType()) || As<ScalarType>(capture->GetType()), capture->span_)
          << "VF captures must be Tiles or Scalars";
      if (As<ScalarType>(capture->GetType())) {
        substitutions[capture.get()] =
            std::make_shared<Var>(capture->name_hint_, capture->GetType(), capture->span_);
      }
    }
    std::vector<VarPtr> params;
    std::vector<ExprPtr> args;
    std::vector<ParamDirection> directions;
    for (const auto& capture : captures) {
      if (!substitutions.count(capture.get())) {
        auto type = CloneTypeWithMemRefAndRemapExprs(
            capture->GetType(), std::nullopt,
            [&](const ExprPtr& expr) { return transform_utils::Substitute(expr, substitutions); });
        substitutions[capture.get()] = std::make_shared<Var>(capture->name_hint_, type, capture->span_);
      }
      params.push_back(AsVarLike(substitutions.at(capture.get())));
      args.push_back(capture);
      directions.push_back(ParamDirection::In);
    }
    std::vector<VarPtr> outputs;
    std::vector<ExprPtr> returns;
    std::vector<TypePtr> return_types;
    for (const auto* raw : operands.var_defs_ordered) {
      auto use = index_.uses.find(raw);
      if (use == index_.uses.end() || (use->second.size() == 1 && use->second.count(scope.get()))) continue;
      auto object = operands.objects.find(raw);
      if (object == operands.objects.end() || !As<TileType>(object->second->GetType())) continue;
      outputs.push_back(object->second);
      available_.insert(raw);
    }
    std::string name;
    do {
      name = function_->name_ + "_vf_" + std::to_string(next_name_++);
    } while (!names_.insert(name).second);
    // Preserve repeated pre-SSA definitions, but give every local binding a
    // helper-owned identity. SSA conversion runs on both functions afterwards.
    for (const auto* raw : operands.var_defs_ordered) {
      if (!substitutions.count(raw) && operands.objects.count(raw)) {
        const auto& var = operands.objects.at(raw);
        // DeepClone creates IterArgs at their loop definition so their seeds
        // are substituted in the enclosing scope. They must remain IterArgs.
        if (As<IterArg>(var)) continue;
        auto type = CloneTypeWithMemRefAndRemapExprs(var->GetType(), std::nullopt, [&](const ExprPtr& expr) {
          return transform_utils::Substitute(expr, substitutions);
        });
        substitutions[raw] = std::make_shared<Var>(var->name_hint_, type, var->span_);
      }
    }
    auto region = DeepClone(scope, substitutions, false).cloned_body;
    for (const auto& output : outputs) {
      returns.push_back(substitutions.at(output.get()));
      return_types.push_back(returns.back()->GetType());
    }
    auto body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{region, std::make_shared<ReturnStmt>(returns, scope->span_)}, scope->span_);
    auto helper = std::make_shared<Function>(name, params, directions, return_types, body, scope->span_,
                                             FunctionType::VF);
    helpers_.push_back(NormalizeHelper(helper));
    // Call result types belong to the caller; helper parameter types contain
    // fresh scalar symbols and must never leak back through the call's type.
    std::vector<TypePtr> caller_return_types;
    caller_return_types.reserve(outputs.size());
    for (const auto& output : outputs) caller_return_types.push_back(output->GetType());
    TypePtr result_type = caller_return_types.size() == 1 ? caller_return_types[0]
                                                          : std::make_shared<TupleType>(caller_return_types);
    if (caller_return_types.empty()) result_type = std::make_shared<UnknownType>();
    auto call =
        std::make_shared<Call>(std::make_shared<GlobalVar>(name), args,
                               std::vector<std::pair<std::string, std::any>>{}, result_type, scope->span_);
    if (outputs.empty()) return std::make_shared<EvalStmt>(call, scope->span_, scope->leading_comments_);
    if (outputs.size() == 1) {
      return std::make_shared<AssignStmt>(outputs[0], call, scope->span_, scope->leading_comments_);
    }
    auto tuple = std::make_shared<Var>(name + "_results", result_type, scope->span_);
    std::vector<StmtPtr> stmts{
        std::make_shared<AssignStmt>(tuple, call, scope->span_, scope->leading_comments_)};
    for (size_t i = 0; i < outputs.size(); ++i) {
      stmts.push_back(std::make_shared<AssignStmt>(
          outputs[i], std::make_shared<TupleGetItemExpr>(tuple, static_cast<int>(i), scope->span_),
          scope->span_));
    }
    return std::make_shared<SeqStmts>(stmts, scope->span_);
  }

 private:
  FunctionPtr function_;
  const RegionIndex& index_;
  std::set<std::string>& names_;
  std::vector<FunctionPtr>& helpers_;
  std::set<const Var*> available_;
  size_t next_name_ = 0;
};
}  // namespace

namespace pass {
Pass OutlineRegScopes() {
  return CreateProgramPass(
      [](const ProgramPtr& program) {
        std::set<std::string> names;
        for (const auto& [_, function] : program->functions_) names.insert(function->name_);
        auto expanded = transform_utils::InlineRegScopeCalls(program);
        auto result = MutableCopy(expanded);
        std::vector<FunctionPtr> helpers;
        for (auto& [_, function] : result->functions_) {
          if (function->func_type_ == FunctionType::VF || !function->body_) continue;
          CheckRegScopes(function);
          // Propagate proven caller constants before captures hide their values
          // behind fresh parameters. Inlining has exposed scalar Out writes,
          // so the pre-SSA guards can reject rebound values. Caller control
          // flow remains intact until its normal pipeline passes.
          function = transform_utils::SimplifyFunction(function, /*constants_only=*/true);
          function = transform_utils::UnrollRegScopeAncestors(function);
          RegionIndex index;
          index.VisitStmt(function->body_);
          Outliner outliner(function, index, names, helpers);
          function = outliner.VisitFunction(function);
        }
        for (const auto& helper : helpers) {
          result->functions_.emplace(std::make_shared<GlobalVar>(helper->name_), helper);
        }
        CheckVFCalls(result);
        return ProgramPtr(helpers.empty() ? expanded : result);
      },
      "OutlineRegScopes", kOutlineRegScopesProperties);
}
}  // namespace pass
}  // namespace pypto::ir
