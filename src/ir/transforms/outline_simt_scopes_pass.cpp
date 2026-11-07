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
#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/simt_analysis.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {

// Index region-local definitions once. The caller walk subsequently rejects
// escapes by identity, including uses before a region and same-spelling Vars.
class RegionIndex : public IRVisitor {
 public:
  std::unordered_set<const Var*> locals;
  bool found = false;

  bool MayRebind(const Var* var, const Stmt* loop) const {
    if (!loop) return false;
    const auto it = rebindings_.find(var);
    if (it == rebindings_.end()) return false;
    const auto& [begin, end] = loop_ranges_.at(loop);
    const auto position = std::lower_bound(it->second.begin(), it->second.end(), begin);
    return position != it->second.end() && *position < end;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    auto call = As<Call>(stmt->value_);
    // A store returning its own destination updates contents, not the handle.
    if (!(call && IsOp(call, "tile.store") && call->args_.size() >= 3 &&
          AsVarLike(call->args_[2]) == stmt->var_)) {
      rebindings_[stmt->var_.get()].push_back(next_binding_++);
    }
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const ForStmtPtr& stmt) override {
    const auto begin = next_binding_;
    IRVisitor::VisitStmt_(stmt);
    loop_ranges_[stmt.get()] = {begin, next_binding_};
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    const auto begin = next_binding_;
    IRVisitor::VisitStmt_(stmt);
    loop_ranges_[stmt.get()] = {begin, next_binding_};
  }
  void VisitStmt_(const SimtScopeStmtPtr& scope) override {
    found = true;
    var_collectors::VarDefUseCollector collector;
    collector.VisitStmt(scope->body_);
    for (const auto* var : collector.var_defs) {
      CHECK_SPAN(!As<TileType>(var->GetType()) && !As<TensorType>(var->GetType()), var->span_)
          << "SIMT helpers cannot allocate or rebind storage handles";
    }
    for (const auto& var : scope->private_scalars_) collector.var_defs.erase(var.get());
    locals.insert(collector.var_defs.begin(), collector.var_defs.end());
  }

 private:
  // Assignment positions let nested loops query rebindings without rescanning
  // their bodies or copying the full dataflow state at each loop boundary.
  size_t next_binding_ = 0;
  std::unordered_map<const Var*, std::vector<size_t>> rebindings_;
  std::unordered_map<const Stmt*, std::pair<size_t, size_t>> loop_ranges_;
};

class RegionControlChecker : public IRVisitor {
 protected:
  void VisitStmt_(const ForStmtPtr& stmt) override {
    ++loop_depth_;
    ++yield_depth_;
    IRVisitor::VisitStmt_(stmt);
    --yield_depth_;
    --loop_depth_;
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    ++loop_depth_;
    ++yield_depth_;
    IRVisitor::VisitStmt_(stmt);
    --yield_depth_;
    --loop_depth_;
  }
  void VisitStmt_(const IfStmtPtr& stmt) override {
    ++yield_depth_;
    IRVisitor::VisitStmt_(stmt);
    --yield_depth_;
  }
  void VisitStmt_(const YieldStmtPtr& stmt) override {
    CHECK_SPAN(yield_depth_ != 0, stmt->span_) << "yield cannot cross a SIMT region boundary";
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const BreakStmtPtr& stmt) override {
    CHECK_SPAN(loop_depth_ != 0, stmt->span_) << "break cannot cross a SIMT region boundary";
  }
  void VisitStmt_(const ContinueStmtPtr& stmt) override {
    CHECK_SPAN(loop_depth_ != 0, stmt->span_) << "continue cannot cross a SIMT region boundary";
  }
  void VisitStmt_(const ReturnStmtPtr& stmt) override {
    CHECK_SPAN(false, stmt->span_) << "return cannot cross a SIMT region boundary";
  }
  void VisitStmt_(const SimtScopeStmtPtr& stmt) override {
    CHECK_SPAN(false, stmt->span_) << "Nested simt_scope is not supported";
  }

 private:
  size_t loop_depth_ = 0;
  size_t yield_depth_ = 0;
};

// A conservative storage-alias partition is built once, before mutation. This
// handles pre-SSA reassignments and branch/loop joins without choosing a single
// reaching definition. Union by rank and path compression keep it near-linear.
class StoragePermissions : public IRVisitor {
 public:
  explicit StoragePermissions(const FunctionPtr& function) {
    for (size_t i = 0; i < function->params_.size(); ++i) {
      const auto& param = function->params_[i];
      if (!IsStorage(param)) continue;
      const auto direction = function->param_directions_[i];
      auto& root = nodes_[param.get()];
      root.readonly = direction == ParamDirection::In;
      root.uninitialized = direction == ParamDirection::Out;
    }
    VisitStmt(function->body_);
  }

  const Var* Root(const Var* var) {
    auto& node = nodes_[var];
    if (!node.parent) return var;
    return node.parent = Root(node.parent);
  }
  bool Known(const Var* var) { return !nodes_.at(Root(var)).unknown_provenance; }
  bool Readonly(const Var* var) { return nodes_.at(Root(var)).readonly; }
  bool Uninitialized(const Var* var) { return nodes_.at(Root(var)).uninitialized; }

 protected:
  void VisitStmt_(const SimtScopeStmtPtr&) override {}
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (As<TupleType>(stmt->var_->GetType())) {
      auto tuple = ResolveTuple(stmt->value_);
      auto [it, inserted] = tuples_.try_emplace(stmt->var_.get(), tuple);
      if (!inserted && it->second != tuple) it->second = nullptr;
      return;
    }
    if (!IsStorage(stmt->var_)) return;
    if (auto source = AsVarLike(stmt->value_)) {
      Unite(stmt->var_, source);
    } else if (auto projection = As<TupleGetItemExpr>(stmt->value_)) {
      auto tuple = ResolveTuple(projection->tuple_);
      auto source = tuple && projection->index_ >= 0 &&
                            static_cast<size_t>(projection->index_) < tuple->elements_.size()
                        ? AsVarLike(tuple->elements_[projection->index_])
                        : nullptr;
      if (source) {
        Unite(stmt->var_, source);
      } else {
        nodes_[Root(stmt->var_.get())].unknown_provenance = true;
      }
    } else if (auto call = As<Call>(stmt->value_)) {
      if (As<GlobalVar>(call->op_)) nodes_[Root(stmt->var_.get())].unknown_provenance = true;
      if (IsOp(call, "tile.create") || IsOp(call, "tensor.create")) {
        auto& root = nodes_[Root(stmt->var_.get())];
        root.uninitialized = true;
      } else {
        auto alias = ResultAliasedArgIndex(call);
        // Tile arithmetic returns a fresh value even if a later memory planner
        // may reuse an operand. Only views and handle copies inherit permission.
        if (As<TileType>(stmt->var_->GetType())) alias = std::nullopt;
        if (op_predicates::IsBufferAliasingViewOp(call->op_->name_) || IsOp(call, "tensor.slice") ||
            IsOp(call, "tensor.reshape") || IsOp(call, "tensor.view")) {
          alias = 0;
        }
        if (alias && *alias < call->args_.size()) Unite(stmt->var_, AsVarLike(call->args_[*alias]));
      }
    }
  }
  void VisitStmt_(const ForStmtPtr& stmt) override {
    Carries(stmt->iter_args_, stmt->return_vars_);
    IRVisitor::VisitStmt_(stmt);
    YieldRoots(stmt->body_, stmt->return_vars_);
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    Carries(stmt->iter_args_, stmt->return_vars_);
    IRVisitor::VisitStmt_(stmt);
    YieldRoots(stmt->body_, stmt->return_vars_);
  }
  void VisitStmt_(const IfStmtPtr& stmt) override {
    IRVisitor::VisitStmt_(stmt);
    YieldRoots(stmt->then_body_, stmt->return_vars_);
    if (stmt->else_body_) YieldRoots(*stmt->else_body_, stmt->return_vars_);
  }

 private:
  struct Node {
    const Var* parent = nullptr;
    size_t rank = 0;
    bool readonly = false;
    bool uninitialized = false;
    bool unknown_provenance = false;
  };
  MakeTuplePtr ResolveTuple(const ExprPtr& expr) const {
    if (auto tuple = As<MakeTuple>(expr)) return tuple;
    if (auto var = AsVarLike(expr)) {
      auto it = tuples_.find(var.get());
      if (it != tuples_.end()) return it->second;
    }
    if (auto projection = As<TupleGetItemExpr>(expr)) {
      auto var = AsVarLike(projection->tuple_);
      auto it = var ? tuples_.find(var.get()) : tuples_.end();
      if (it != tuples_.end() && it->second && projection->index_ >= 0 &&
          static_cast<size_t>(projection->index_) < it->second->elements_.size()) {
        return As<MakeTuple>(it->second->elements_[projection->index_]);
      }
    }
    return nullptr;
  }
  static bool IsStorage(const VarPtr& var) {
    return As<TileType>(var->GetType()) || As<TensorType>(var->GetType());
  }
  void Unite(const VarPtr& a, const VarPtr& b) {
    if (!a || !b || !IsStorage(a) || !IsStorage(b)) return;
    auto left = Root(a.get());
    auto right = Root(b.get());
    if (left == right) return;
    if (nodes_.at(left).rank < nodes_.at(right).rank) std::swap(left, right);
    auto& parent = nodes_.at(left);
    auto& child = nodes_.at(right);
    child.parent = left;
    parent.rank += parent.rank == child.rank;
    parent.readonly |= child.readonly;
    parent.unknown_provenance |= child.unknown_provenance;
    parent.uninitialized |= child.uninitialized;
  }
  void Carries(const std::vector<IterArgPtr>& args, const std::vector<VarPtr>& results) {
    for (size_t i = 0; i < args.size(); ++i) {
      Unite(args[i], AsVarLike(args[i]->initValue_));
      if (i < results.size()) Unite(results[i], args[i]);
    }
  }
  void YieldRoots(const StmtPtr& body, const std::vector<VarPtr>& results) {
    // Only the direct final yield belongs to this control-flow construct.
    auto tail = body;
    if (auto seq = As<SeqStmts>(body); seq && !seq->stmts_.empty()) {
      tail = seq->stmts_.back();
    }
    if (auto yield = As<YieldStmt>(tail)) {
      for (size_t i = 0; i < results.size() && i < yield->value_.size(); ++i) {
        Unite(results[i], AsVarLike(yield->value_[i]));
      }
    }
  }
  std::unordered_map<const Var*, Node> nodes_;
  std::unordered_map<const Var*, MakeTuplePtr> tuples_;
};

class ScopeOutliner : public IRMutator {
 public:
  ScopeOutliner(FunctionPtr function, const RegionIndex& index, std::set<std::string>& names,
                std::vector<FunctionPtr>& helpers)
      : function_(std::move(function)),
        index_(index),
        names_(names),
        helpers_(helpers),
        permissions_(function_) {
    outline_utils::VarCollector variables;
    variables.VisitStmt(function_->body_);
    for (const auto& param : function_->params_) variables.CollectParameter(param);
    for (const auto& [var, _] : variables.var_objects) {
      binding_indices_.emplace(var, binding_indices_.size());
    }
    binding_capacity_ = std::max(size_t{1}, binding_indices_.size());
    for (size_t i = 0; i < function_->params_.size(); ++i) {
      const auto& param = function_->params_[i];
      if (IsStorage(param)) {
        BindFresh(param, function_->param_directions_[i] != ParamDirection::Out);
      }
    }
  }

 protected:
  ExprPtr VisitExpr_(const VarPtr& var) override {
    CheckEscape(var);
    return IRMutator::VisitExpr_(var);
  }
  ExprPtr VisitExpr_(const IterArgPtr& var) override {
    CheckEscape(var);
    return IRMutator::VisitExpr_(var);
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& stmt) override {
    auto result = IRMutator::VisitStmt_(stmt);
    if (auto call = As<Call>(stmt->value_)) RecordFullInitialization(call);
    if (IsStorage(stmt->var_)) {
      auto value = ResolveStorage(stmt->value_);
      if (value.storage) {
        Bind(stmt->var_, value);
      } else {
        auto call = As<Call>(stmt->value_);
        const bool fresh_tile = call && As<TileType>(stmt->var_->GetType()) && !IsOp(call, "tile.create") &&
                                !As<GlobalVar>(call->op_);
        BindFresh(stmt->var_, fresh_tile && InitializedExpression(call));
      }
    }
    return result;
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& stmt) override {
    auto result = IRMutator::VisitStmt_(stmt);
    if (auto call = As<Call>(stmt->expr_)) RecordFullInitialization(call);
    return result;
  }
  StmtPtr VisitStmt_(const IfStmtPtr& stmt) override {
    auto result = MutableCopy(stmt);
    result->condition_ = VisitExpr(stmt->condition_);
    auto incoming = bindings_;
    result->then_body_ = VisitConditionalBody(stmt->then_body_);
    auto then_bindings = bindings_;
    bindings_ = incoming;
    if (stmt->else_body_) result->else_body_ = VisitConditionalBody(*stmt->else_body_);
    bindings_ = JoinBindings(incoming, then_bindings, bindings_);
    return result;
  }
  StmtPtr VisitStmt_(const ForStmtPtr& stmt) override { return VisitLoop(stmt); }
  StmtPtr VisitStmt_(const WhileStmtPtr& stmt) override { return VisitLoop(stmt); }
  StmtPtr VisitStmt_(const RegScopeStmtPtr& scope) override {
    ++reg_depth_;
    auto result = IRMutator::VisitStmt_(scope);
    --reg_depth_;
    return result;
  }
  StmtPtr VisitStmt_(const SimtScopeStmtPtr& scope) override {
    CHECK_SPAN(function_->func_type_ == FunctionType::AIV, scope->span_)
        << "simt_scope requires an explicit AIV function";
    CHECK_SPAN(reg_depth_ == 0, scope->span_) << "simt_scope is not allowed inside reg_scope";
    RegionControlChecker control;
    control.VisitStmt(scope->body_);
    var_collectors::VarDefUseCollector uses;
    uses.VisitStmt(scope->body_);
    outline_utils::VarCollector objects;
    objects.VisitStmt(scope->body_);
    std::unordered_set<const Var*> private_scalars;
    for (const auto& var : scope->private_scalars_) private_scalars.insert(var.get());
    std::vector<VarPtr> captures;
    for (const auto* var : uses.var_uses_ordered) {
      if (!uses.var_defs.count(var) || private_scalars.count(var)) {
        captures.push_back(objects.var_objects.at(var));
      }
    }
    auto probe = std::make_shared<Function>(
        "simt_capture_analysis", captures, std::vector<ParamDirection>(captures.size(), ParamDirection::In),
        std::vector<TypePtr>{}, scope->body_, scope->span_, FunctionType::SIMT);
    const auto accesses = AnalyzeSimtAccess(probe);
    std::vector<VarPtr> params;
    std::vector<ExprPtr> args;
    std::vector<VarPtr> results;
    std::vector<int> result_indices;
    std::vector<ParamDirection> directions;
    std::unordered_map<const Var*, ExprPtr> substitutions;
    for (size_t i = 0; i < captures.size(); ++i) {
      const auto& capture = captures[i];
      CHECK_SPAN(permissions_.Known(capture.get()), capture->span_)
          << "Cannot prove SIMT capture permissions through a call result or rebound tuple; "
             "capture the original named storage";
      if (accesses[i].writes) {
        CHECK_SPAN(!permissions_.Readonly(capture.get()), capture->span_)
            << "SIMT store cannot write an enclosing In parameter or its view; declare InOut";
      }
      CHECK_SPAN(IsInitialized(capture), capture->span_)
          << "SIMT capture requires initialized storage; use InOut or initialize Out before the region";
      auto param = std::make_shared<Var>(capture->name_hint_, capture->GetType(), capture->span_);
      params.push_back(param);
      args.push_back(capture);
      substitutions[capture.get()] = param;
      directions.push_back(accesses[i].writes ? ParamDirection::InOut : ParamDirection::In);
      if (accesses[i].writes) {
        results.push_back(capture);
        result_indices.push_back(static_cast<int>(i));
      }
    }
    for (const auto& var : scope->private_scalars_) {
      if (!substitutions.count(var.get())) {
        substitutions[var.get()] = std::make_shared<Var>(var->name_hint_, var->GetType(), var->span_);
      }
    }
    std::string name;
    do {
      name = function_->name_ + "_simt_" + std::to_string(next_name_++);
    } while (!names_.insert(name).second);
    // Pre-SSA assignments share definition identities. Preserve those identities
    // within the helper while replacing every captured binding with a parameter.
    auto body = DeepClone(scope->body_, substitutions, false).cloned_body;
    helpers_.push_back(std::make_shared<Function>(name, params, directions, std::vector<TypePtr>{}, body,
                                                  scope->span_, FunctionType::SIMT));
    auto launch =
        std::make_shared<SimtLaunchStmt>(std::make_shared<GlobalVar>(name), scope->block_dims_, args, results,
                                         result_indices, scope->span_, scope->leading_comments_);
    InferSimtResources(launch);
    return launch;
  }

 private:
  // Permissions are may-alias facts; initialization needs the reaching storage
  // value. Each fresh Tile/allocation gets a separate identity, while copies
  // and views share their actual source. A branch-local identity/proof expires
  // at its join. We deliberately do not infer a reaching alias across joins.
  // A persistent segment tree snapshots sibling environments in O(1), with
  // O(log N) binding updates/lookups. Joins share unchanged subtrees and visit
  // only overlapping changed paths; they do not rescan an incoming environment.
  // Scope stamps expire branch-local values and initialization proofs at joins.
  struct Storage {
    TypePtr type;
    std::optional<size_t> initialized_scope;
    bool loop_carry = false;
    bool requires_initialized_backedge = false;
  };
  struct Binding {
    std::shared_ptr<Storage> storage;
    bool whole = false;
    size_t scope = 0;
    size_t loop_depth = 0;
  };
  struct BindingNode;
  using BindingRoot = std::shared_ptr<const BindingNode>;
  struct BindingNode {
    BindingRoot left;
    BindingRoot right;
    std::optional<Binding> value;
  };
  std::optional<Binding> LookupBinding(const Var* var) const {
    auto index = binding_indices_.find(var);
    if (index == binding_indices_.end()) return std::nullopt;
    auto node = bindings_;
    size_t begin = 0;
    size_t end = binding_capacity_;
    while (node && end - begin > 1) {
      const auto middle = begin + (end - begin) / 2;
      if (index->second < middle) {
        node = node->left;
        end = middle;
      } else {
        node = node->right;
        begin = middle;
      }
    }
    return node ? node->value : std::nullopt;
  }
  static BindingRoot UpdateBinding(const BindingRoot& root, size_t begin, size_t end, size_t index,
                                   const Binding& binding) {
    auto node = root ? std::make_shared<BindingNode>(*root) : std::make_shared<BindingNode>();
    if (end - begin == 1) {
      node->value = binding;
    } else {
      const auto middle = begin + (end - begin) / 2;
      if (index < middle) {
        node->left = UpdateBinding(node->left, begin, middle, index, binding);
      } else {
        node->right = UpdateBinding(node->right, middle, end, index, binding);
      }
    }
    return node;
  }
  static BindingRoot JoinBindings(const BindingRoot& incoming, const BindingRoot& left,
                                  const BindingRoot& right) {
    if (left == incoming) return right;
    if (right == incoming || left == right) return left;
    // Both sides replaced this leaf. Either expired branch binding records
    // that the reaching value is unknown; neither supplies a post-join proof.
    if (left->value) return left;
    auto node = std::make_shared<BindingNode>();
    node->left = JoinBindings(incoming ? incoming->left : nullptr, left->left, right->left);
    node->right = JoinBindings(incoming ? incoming->right : nullptr, left->right, right->right);
    return node;
  }
  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& stmt) {
    std::vector<bool> seed_initialized;
    for (const auto& arg : stmt->iter_args_) {
      seed_initialized.push_back(InitializedExpression(arg->initValue_));
    }
    const auto parent_scope = EnterConditional();
    loop_stack_.push_back(stmt.get());
    std::vector<std::shared_ptr<Storage>> carries;
    for (size_t i = 0; i < stmt->iter_args_.size(); ++i) {
      const auto& arg = stmt->iter_args_[i];
      if (!IsStorage(arg)) {
        carries.push_back(nullptr);
        continue;
      }
      // A carry has an initialized-value invariant, not a definite alias to
      // its seed: subsequent iterations may yield a different allocation.
      BindFresh(arg, seed_initialized[i]);
      auto storage = ResolveStorage(arg).storage;
      storage->loop_carry = true;
      carries.push_back(std::move(storage));
    }
    auto result = IRMutator::VisitStmt_(stmt);
    auto tail = stmt->body_;
    if (auto seq = As<SeqStmts>(tail); seq && !seq->stmts_.empty()) {
      tail = seq->stmts_.back();
    }
    auto yield = As<YieldStmt>(tail);
    // Evaluate every backedge before checking requirements: a later yield may
    // consume an earlier carry's proof (including through a Tile expression).
    // Two linear passes prevent parameter order from hiding that dependency.
    std::vector<bool> backedge_initialized;
    for (size_t i = 0; i < carries.size(); ++i) {
      const ExprPtr value = yield && i < yield->value_.size() ? yield->value_[i] : stmt->iter_args_[i];
      backedge_initialized.push_back(InitializedExpression(value));
    }
    std::vector<bool> result_initialized;
    for (size_t i = 0; i < carries.size(); ++i) {
      CHECK_SPAN(!carries[i] || !carries[i]->requires_initialized_backedge || backedge_initialized[i],
                 stmt->span_)
          << "SIMT loop carry requires initialized storage on every iteration";
      result_initialized.push_back(seed_initialized[i] && backedge_initialized[i]);
    }
    loop_stack_.pop_back();
    LeaveConditional(parent_scope);
    for (size_t i = 0; i < stmt->return_vars_.size() && i < result_initialized.size(); ++i) {
      if (IsStorage(stmt->return_vars_[i])) BindFresh(stmt->return_vars_[i], result_initialized[i]);
    }
    return result;
  }
  static bool IsStorage(const VarPtr& var) {
    return As<TileType>(var->GetType()) || AsTensorTypeLike(var->GetType());
  }
  Binding ResolveStorage(const ExprPtr& expr) {
    if (auto var = AsVarLike(expr)) {
      auto binding = LookupBinding(var.get());
      if (binding && active_scopes_[binding->scope] &&
          (binding->loop_depth == loop_stack_.size() ||
           !index_.MayRebind(var.get(), loop_stack_[binding->loop_depth]))) {
        return *binding;
      }
    } else if (auto call = As<Call>(expr)) {
      auto alias = ResultAliasedArgIndex(call);
      if (As<TileType>(call->GetType())) alias = std::nullopt;
      if (op_predicates::IsBufferAliasingViewOp(call->op_->name_) || IsOp(call, "tensor.slice") ||
          IsOp(call, "tensor.reshape") || IsOp(call, "tensor.view")) {
        alias = 0;
      }
      if (alias && *alias < call->args_.size()) {
        auto source = ResolveStorage(call->args_[*alias]);
        // Only unchanged storage geometry can prove an entire allocation
        // initialized. A full write to a subview is not a full root write.
        source.whole = source.whole && structural_equal(call->GetType(), call->args_[*alias]->GetType());
        return source;
      }
    }
    return {};
  }
  bool IsInitialized(const VarPtr& var) {
    if (!IsStorage(var)) return true;
    auto binding = ResolveStorage(var);
    if (binding.storage) {
      const auto scope = binding.storage->initialized_scope;
      const bool initialized = scope && active_scopes_[*scope];
      if (initialized && binding.storage->loop_carry) binding.storage->requires_initialized_backedge = true;
      return initialized;
    }
    // An expired or loop-carried binding cannot reuse the first iteration's
    // proof: a later assignment may replace its storage before the backedge.
    if (LookupBinding(var.get())) return false;
    // At joins where no reaching value is known, retain only the conservative
    // guarantee that no possible source is Out or an uninitialized allocation.
    return !permissions_.Uninitialized(var.get());
  }
  bool InitializedExpression(const ExprPtr& expr) {
    // Seeds/yields precede FlattenCallExpr. Walk each expression once, checking
    // nested allocations as well as named operands; tile.full and arithmetic
    // produce initialized values only when their storage inputs are initialized.
    class Checker : public IRVisitor {
     public:
      explicit Checker(ScopeOutliner& owner) : owner_(owner) {}
      bool initialized = true;

     protected:
      void VisitExpr_(const VarPtr& var) override { initialized &= owner_.IsInitialized(var); }
      void VisitExpr_(const IterArgPtr& var) override { initialized &= owner_.IsInitialized(var); }
      void VisitExpr_(const TupleGetItemExprPtr& projection) override {
        // A tuple may retain an older storage handle than its source Var's
        // current binding. Until tuple values have reaching-storage identities,
        // neither a nonstorage tuple Var nor its original spelling is proof.
        if (As<TileType>(projection->GetType()) || AsTensorTypeLike(projection->GetType())) {
          initialized = false;
        }
        IRVisitor::VisitExpr_(projection);
      }
      void VisitExpr_(const CallPtr& call) override {
        if (As<TileType>(call->GetType()) || AsTensorTypeLike(call->GetType())) {
          if (As<GlobalVar>(call->op_) || IsOp(call, "tile.create") || IsOp(call, "tensor.create") ||
              (!As<TileType>(call->GetType()) && !ResultAliasedArgIndex(call))) {
            initialized = false;
          }
        }
        IRVisitor::VisitExpr_(call);
      }

     private:
      ScopeOutliner& owner_;
    } checker(*this);
    checker.VisitExpr(expr);
    return checker.initialized;
  }
  void Bind(const VarPtr& var, Binding binding) {
    auto previous = ResolveStorage(var);
    if (previous.storage == binding.storage && previous.whole == binding.whole) return;
    binding.scope = current_scope_;
    binding.loop_depth = loop_stack_.size();
    bindings_ = UpdateBinding(bindings_, 0, binding_capacity_, binding_indices_.at(var.get()), binding);
  }
  void BindFresh(const VarPtr& var, bool initialized) {
    auto storage = std::make_shared<Storage>();
    storage->type = var->GetType();
    if (initialized) storage->initialized_scope = current_scope_;
    Bind(var, {std::move(storage), true, current_scope_});
  }
  size_t EnterConditional() {
    const auto parent = current_scope_;
    current_scope_ = active_scopes_.size();
    active_scopes_.push_back(true);
    return parent;
  }
  void LeaveConditional(size_t parent) {
    active_scopes_[current_scope_] = false;
    current_scope_ = parent;
  }
  // A full, unmasked tile.store proves only its definite destination initialized.
  // Existing unconditional proofs are retained when a conditional store runs.
  static bool SameShape(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (!structural_equal(a[i], b[i])) return false;
    }
    return true;
  }
  void RecordFullInitialization(const CallPtr& call) {
    if (!IsOp(call, "tile.store") || call->args_.size() < 3) return;
    auto input = AsVarLike(call->args_[0]);
    if (!input || !IsInitialized(input)) return;
    auto source = As<TileType>(call->args_[0]->GetType());
    auto offsets = As<MakeTuple>(call->args_[1]);
    auto destination = AsVarLike(call->args_[2]);
    auto tensor = destination ? As<TensorType>(destination->GetType()) : nullptr;
    if (!source || !offsets || !tensor || source->shape_.size() != tensor->shape_.size()) return;
    if (!SameShape(source->shape_, tensor->shape_)) return;
    if (source->tile_view_ && !source->tile_view_->valid_shape.empty() &&
        !SameShape(source->tile_view_->valid_shape, source->shape_)) {
      return;
    }
    for (const auto& offset : offsets->elements_) {
      auto constant = As<ConstInt>(offset);
      if (!constant || constant->value_ != 0) return;
    }
    auto binding = ResolveStorage(destination);
    if (!binding.storage || !binding.whole ||
        !structural_equal(destination->GetType(), binding.storage->type)) {
      return;
    }
    if (!IsInitialized(destination)) binding.storage->initialized_scope = current_scope_;
  }
  StmtPtr VisitConditionalBody(const StmtPtr& body) {
    const auto parent_scope = EnterConditional();
    auto result = VisitStmt(body);
    LeaveConditional(parent_scope);
    return result;
  }
  void CheckEscape(const VarPtr& var) const {
    CHECK_SPAN(!index_.locals.count(var.get()), var->span_)
        << "SIMT thread-private scalar cannot escape its region";
  }
  FunctionPtr function_;
  const RegionIndex& index_;
  std::set<std::string>& names_;
  std::vector<FunctionPtr>& helpers_;
  StoragePermissions permissions_;
  std::unordered_map<const Var*, size_t> binding_indices_;
  size_t binding_capacity_ = 1;
  BindingRoot bindings_;
  std::vector<bool> active_scopes_{true};
  size_t current_scope_ = 0;
  std::vector<const Stmt*> loop_stack_;
  size_t next_name_ = 0;
  size_t reg_depth_ = 0;
};

ProgramPtr Outline(const ProgramPtr& program) {
  std::set<std::string> names;
  for (const auto& [_, function] : program->functions_) names.insert(function->name_);
  std::vector<FunctionPtr> helpers;
  auto result = MutableCopy(program);
  for (auto& [_, function] : result->functions_) {
    RegionIndex index;
    index.VisitStmt(function->body_);
    if (!index.found) continue;
    ScopeOutliner outliner(function, index, names, helpers);
    function = outliner.VisitFunction(function);
  }
  for (const auto& helper : helpers) {
    result->functions_.emplace(std::make_shared<GlobalVar>(helper->name_), helper);
  }
  // Reuse the canonical helper legality contract, even when pass verification
  // is disabled. No unsupported scalar op or storage operation can bypass it.
  if (!helpers.empty()) {
    std::vector<Diagnostic> diagnostics;
    CreateSimtValidPropertyVerifier()->Verify(result, diagnostics);
    for (const auto& diagnostic : diagnostics) {
      CHECK_SPAN(diagnostic.severity != DiagnosticSeverity::Error, diagnostic.span) << diagnostic.message;
    }
  }
  return helpers.empty() ? program : result;
}
}  // namespace

namespace pass {
Pass OutlineSimtScopes() {
  return CreateProgramPass(Outline, "OutlineSimtScopes", kOutlineSimtScopesProperties);
}
}  // namespace pass
}  // namespace pypto::ir
