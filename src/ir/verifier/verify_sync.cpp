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
 * @file verify_sync.cpp
 * @brief Verify logical completion values without assigning hardware resources.
 *
 * One lexical statement walk records token lifetimes and expression use sites.
 * Expressions and type metadata are expanded once, then use intervals and a
 * fixed set of execution contexts propagate through their DAG in reverse
 * postorder. This keeps verification O(N log N) in unique nodes and edges,
 * without copying scopes or expanding shared expressions into trees. The
 * verifier does not prove target-specific event implementability.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/type_metadata.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/sync.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {

class LogicalSyncChecker : public IRVisitor {
 public:
  LogicalSyncChecker(FunctionPtr function, std::vector<Diagnostic>& diagnostics)
      : function_(std::move(function)), diagnostics_(diagnostics) {}

  /// Check signatures before the body so malformed incoming tokens never gain ownership.
  void Run() {
    for (const auto& parameter : function_->params_) {
      // Storage descriptors can themselves be parameters; their fields are
      // metadata on the value, rather than on its otherwise opaque type.
      VisitExpr(parameter);
      if (ContainsToken(parameter->GetType())) {
        Error("SyncToken cannot cross a function parameter boundary, including inside a tuple",
              parameter->span_);
      }
    }
    for (const auto& type : function_->return_types_) {
      VisitTypeMetadata(type);
      if (ContainsToken(type)) {
        Error("SyncToken cannot cross a function return boundary, including inside a tuple",
              function_->span_);
      }
    }
    VisitMetadata(function_->attrs_);
    if (function_->body_) VisitStmt(function_->body_);
    CheckExpressionUses();
    if (has_logical_sync_ && nonlocal_call_) {
      Error(
          "Logical synchronization requires a leaf AIC/AIV function; inline function calls and "
          "eliminate Submit before inserting synchronization",
          nonlocal_call_->span_);
    }
  }

  /// Record every use, but expand each expression/type DAG node only once.
  void VisitExpr(const ExprPtr& expression) override { VisitExpressionUse(expression, false); }

  /// Scope attributes carry metadata, never a second channel for token ownership.
  void VisitStmt(const StmtPtr& statement) override {
    if (!statement) return;
    ++position_;
    if (auto scope = As<ScopeStmt>(statement)) VisitMetadata(scope->attrs_);
    IRVisitor::VisitStmt(statement);
  }

 protected:
  // Visibility is checked after propagating all uses. Type metadata is already
  // handled by VisitExpr, so the base visitor must not rescan tensor shapes.
  void VisitVarLike_(const VarPtr& /*variable*/) override {}

  // An IterArg use refers to the current iteration value. Its initializer is
  // evaluated once at the loop boundary, never recursively at each use.
  void VisitExpr_(const IterArgPtr& /*argument*/) override {}

  /// Direct storage descriptors obey the same rules as type-owned descriptors.
  void VisitExpr_(const MemRefPtr& memref) override { VisitStorageMetadata(memref); }
  void VisitExpr_(const WindowBufferPtr& window) override { VisitStorageMetadata(window); }

  void VisitExpr_(const CallPtr& call) override {
    const auto kind = GetLogicalSyncKind(call->op_);
    if (kind != LogicalSyncKind::None) {
      has_logical_sync_ = true;
      RevalidateCall(call);
    } else {
      RejectCallTokens(call->args_, call->GetType(), call->span_);
      if (IsA<GlobalVar>(call->op_)) nonlocal_call_ = call;
    }
    for (const auto& argument : call->args_) VisitExpr(argument);
    VisitMetadata(call->attrs_);
    VisitMetadata(call->kwargs_);
  }

  void VisitExpr_(const SubmitPtr& submit) override {
    if (GetLogicalSyncKind(submit->op_) != LogicalSyncKind::None) {
      Error("Logical synchronization operations require Call; Submit launches a task", submit->span_);
    }
    RejectCallTokens(submit->args_, submit->GetType(), submit->span_);
    nonlocal_call_ = submit;
    IRVisitor::VisitExpr_(submit);
    VisitMetadata(submit->kwargs_);
  }

  /// VisitStmt already checks all scope metadata using the shared attr walker.
  [[nodiscard]] bool ShouldVisitScopeAttr(const std::string& /*key*/) const override { return false; }

  void VisitBinaryExpr_(const BinaryExprPtr& expression) override {
    if (ContainsToken(expression->left_->GetType()) || ContainsToken(expression->right_->GetType()) ||
        ContainsToken(expression->GetType())) {
      Error("SyncToken cannot be used by scalar arithmetic or comparison", expression->span_);
    }
    IRVisitor::VisitBinaryExpr_(expression);
  }

  void VisitUnaryExpr_(const UnaryExprPtr& expression) override {
    if (ContainsToken(expression->operand_->GetType()) || ContainsToken(expression->GetType())) {
      Error("SyncToken cannot be cast or used by a scalar unary operation", expression->span_);
    }
    IRVisitor::VisitUnaryExpr_(expression);
  }

  void VisitStmt_(const AssignStmtPtr& statement) override {
    const auto previous = assignment_root_;
    assignment_root_ = As<Call>(statement->value_).get();
    VisitExpr(statement->value_);
    assignment_root_ = previous;
    CheckMatchingTypes(statement->var_->GetType(), statement->value_->GetType(), statement->span_);
    Define(statement->var_);
  }

  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto previous = evaluation_root_;
    evaluation_root_ = As<Call>(statement->expr_).get();
    VisitExpr(statement->expr_);
    evaluation_root_ = previous;
  }

  void VisitStmt_(const ReturnStmtPtr& statement) override {
    for (const auto& value : statement->value_) {
      if (ContainsToken(value->GetType())) {
        Error("SyncToken cannot escape through a function return", statement->span_);
      }
      VisitExpr(value);
    }
  }

  void VisitStmt_(const IfStmtPtr& statement) override {
    VisitExpr(statement->condition_);
    RejectConditionToken(statement->condition_);
    VisitBranch(statement->then_body_, statement->return_vars_);
    if (statement->else_body_) {
      VisitBranch(*statement->else_body_, statement->return_vars_);
    } else if (AnyToken(statement->return_vars_)) {
      Error("IfStmt with SyncToken results requires both branches and matching terminal yields",
            statement->span_);
    }
    for (const auto& result : statement->return_vars_) Define(result);
  }

  void VisitStmt_(const ForStmtPtr& statement) override {
    VisitExpr(statement->start_);
    VisitExpr(statement->stop_);
    VisitExpr(statement->step_);
    RejectConditionToken(statement->start_);
    RejectConditionToken(statement->stop_);
    RejectConditionToken(statement->step_);
    VisitMetadata(statement->attrs_);
    VisitLoop(statement->iter_args_, statement->return_vars_, statement->body_, nullptr,
              statement->loop_var_);
  }

  void VisitStmt_(const WhileStmtPtr& statement) override {
    VisitLoop(statement->iter_args_, statement->return_vars_, statement->body_, statement->condition_);
  }

  void VisitStmt_(const YieldStmtPtr& statement) override {
    bool token_yield = false;
    for (const auto& value : statement->value_) {
      token_yield = token_yield || ContainsToken(value->GetType());
      VisitExpr(value);
    }
    if (!token_yield && !AnyToken(yield_results_)) return;
    if (statement.get() != terminal_yield_) {
      Error("SyncToken must be yielded at the terminal position of its owning If/For/While region",
            statement->span_);
      return;
    }
    if (statement->value_.size() != yield_results_.size()) {
      Error("SyncToken control-flow yield/result count mismatch", statement->span_);
      return;
    }
    for (size_t index = 0; index < statement->value_.size(); ++index) {
      CheckMatchingTypes(statement->value_[index]->GetType(), yield_results_[index]->GetType(),
                         statement->span_);
    }
  }

  void VisitStmt_(const RegScopeStmtPtr& statement) override {
    const bool previous_vf = in_vf_;
    const auto checkpoint = introduced_.size();
    in_vf_ = true;
    IRVisitor::VisitStmt_(statement);
    Restore(checkpoint);
    in_vf_ = previous_vf;
  }

  // These scopes change dispatch ownership. An enclosing AIC/AIV annotation
  // must not authorize logical tokens nested in a different execution domain.
  void VisitStmt_(const InCoreScopeStmtPtr& statement) override { VisitBlockedScope(statement); }
  void VisitStmt_(const ClusterScopeStmtPtr& statement) override { VisitBlockedScope(statement); }
  void VisitStmt_(const HierarchyScopeStmtPtr& statement) override { VisitBlockedScope(statement); }
  void VisitStmt_(const SpmdScopeStmtPtr& statement) override { VisitBlockedScope(statement); }
  void VisitStmt_(const RuntimeScopeStmtPtr& statement) override { VisitBlockedScope(statement); }
  void VisitStmt_(const GraphScopeStmtPtr& statement) override { VisitBlockedScope(statement); }
  void VisitStmt_(const CommDomainScopeStmtPtr& statement) override { VisitBlockedScope(statement); }

 private:
  static constexpr size_t kNoUse = std::numeric_limits<size_t>::max();
  static constexpr uint8_t kVFContext = 1;
  static constexpr uint8_t kBlockedContext = 2;
  static constexpr uint8_t kMetadataContext = 4;

  /// DFS statement positions give each lexical token binding one half-open lifetime.
  struct UseRange {
    size_t first;
    size_t last;
  };

  struct UseEdge {
    size_t child;
    // Ordinary operands inherit the parent's execution context. Entering type,
    // storage, or attribute metadata additionally sets the metadata context bit.
    bool metadata;
  };

  struct UseRecord {
    ExprPtr expression;  // Null for metadata-only type/storage nodes.
    UseRange range;
    uint8_t contexts;  // Bit i records an occurrence with the three context flags i.
    std::vector<UseEdge> children;
    bool non_assignment = false;
    bool non_evaluation = false;
  };

  /// Cache structure/schema independently of each occurrence's statement and scope.
  void VisitExpressionUse(const ExprPtr& expression, bool metadata) {
    if (!expression) return;
    const auto [id, inserted] = RecordUse(expression_uses_, expression.get(), expression, metadata);
    auto& use = uses_[id];
    use.non_assignment = use.non_assignment || use_parent_ != kNoUse || expression.get() != assignment_root_;
    use.non_evaluation = use.non_evaluation || use_parent_ != kNoUse || expression.get() != evaluation_root_;
    if (!inserted) return;
    const auto previous_parent = std::exchange(use_parent_, id);
    VisitTypeMetadata(expression->GetType());
    IRVisitor::VisitExpr(expression);
    use_postorder_.push_back(id);
    use_parent_ = previous_parent;
  }

  /// Add an occurrence and, only while expanding its parent, one direct DAG edge.
  template <typename Node>
  std::pair<size_t, bool> RecordUse(std::unordered_map<const Node*, size_t>& index, const Node* node,
                                    const ExprPtr& expression, bool metadata) {
    const auto [entry, inserted] = index.emplace(node, uses_.size());
    const auto id = entry->second;
    const auto contexts = CurrentContexts();
    if (inserted) {
      uses_.push_back({expression, {position_, position_}, contexts, {}});
    } else {
      auto& use = uses_[id];
      use.range.first = std::min(use.range.first, position_);
      use.range.last = std::max(use.range.last, position_);
      use.contexts |= contexts;
    }
    if (use_parent_ != kNoUse) uses_[use_parent_].children.push_back({id, metadata});
    return {id, inserted};
  }

  /// There are only eight combinations, independent of lexical nesting depth.
  [[nodiscard]] uint8_t CurrentContexts() const {
    const uint8_t flags =
        (in_vf_ ? kVFContext : 0) | (blocked_ ? kBlockedContext : 0) | (in_metadata_ ? kMetadataContext : 0);
    return static_cast<uint8_t>(uint8_t{1} << flags);
  }

  static bool HasContext(uint8_t contexts, uint8_t flag, bool present = true) {
    for (uint8_t flags = 0; flags < 8; ++flags) {
      if ((contexts & (uint8_t{1} << flags)) && static_cast<bool>(flags & flag) == present) return true;
    }
    return false;
  }

  /// Extremal use positions suffice because every valid binding has one contiguous lifetime.
  /// Contexts propagate separately: metadata/VF escapes must remain visible even
  /// when their expression was first expanded in an otherwise legal outer use.
  void CheckExpressionUses() {
    for (auto it = use_postorder_.rbegin(); it != use_postorder_.rend(); ++it) {
      const auto& source = uses_[*it];
      for (const auto& edge : source.children) {
        auto& target = uses_[edge.child];
        target.range.first = std::min(target.range.first, source.range.first);
        target.range.last = std::max(target.range.last, source.range.last);
        for (uint8_t flags = 0; flags < 8; ++flags) {
          if (source.contexts & (uint8_t{1} << flags)) {
            const auto child_flags = flags | (edge.metadata ? kMetadataContext : 0);
            target.contexts |= uint8_t{1} << child_flags;
          }
        }
      }
    }
    for (const auto& use : uses_) {
      const auto& expression = use.expression;
      if (!expression) continue;
      if (ContainsToken(expression->GetType())) {
        CheckOuterContext(expression->span_, use.contexts);
        if (HasContext(use.contexts, kMetadataContext)) {
          Error("SyncToken cannot escape through IR metadata", expression->span_);
        }
        if (auto variable = AsVarLike(expression)) {
          const auto lifetime = lifetimes_.find(variable.get());
          if (lifetime == lifetimes_.end() || use.range.first < lifetime->second.first ||
              use.range.last >= lifetime->second.last) {
            Error("SyncToken value '" + variable->name_hint_ +
                      "' must be defined before use in the same lexical scope; select branch results or "
                      "carry loop values explicitly",
                  variable->span_);
          }
        }
      }
      if (auto call = As<Call>(expression)) {
        const auto kind = GetLogicalSyncKind(call->op_);
        if (kind == LogicalSyncKind::VFBarrier) {
          if (function_->func_type_ != FunctionType::AIV || HasContext(use.contexts, kVFContext, false) ||
              HasContext(use.contexts, kBlockedContext)) {
            Error("sync.vf_barrier requires a RegScope in a specialized AIV function", call->span_);
          }
        } else if (kind != LogicalSyncKind::None) {
          CheckOuterContext(call->span_, use.contexts);
        }
        if (kind == LogicalSyncKind::Record && use.non_assignment) {
          Error("sync.record requires its own AssignStmt; nested or bare records have no capture position",
                call->span_);
        }
        if ((kind == LogicalSyncKind::Wait || kind == LogicalSyncKind::VFBarrier) && use.non_evaluation) {
          Error("sync.wait and sync.vf_barrier must be standalone EvalStmt operations", call->span_);
        }
      }
    }
  }

  /// Intern token topology once per Type, ignoring data metadata checked by TypeChecked.
  /// ArrayType has a scalar dtype rather than an element Type, so it cannot hold
  /// a token type. Token values passed to array operations are rejected as calls.
  size_t TokenShape(const TypePtr& type) {
    if (!type) return 0;
    if (auto found = token_shapes_.find(type.get()); found != token_shapes_.end()) {
      return found->second;
    }
    std::vector<size_t> key;
    if (auto token = As<SyncTokenType>(type)) {
      key = {0, static_cast<size_t>(token->pipe_)};
    } else if (auto tuple = As<TupleType>(type)) {
      key.reserve(tuple->types_.size() + 1);
      key.push_back(1);
      bool contains = false;
      for (const auto& element : tuple->types_) {
        const auto shape = TokenShape(element);
        key.push_back(shape);
        contains = contains || shape != 0;
      }
      if (!contains) key.clear();
    }
    size_t shape = 0;
    if (!key.empty()) {
      auto found = interned_shapes_.emplace(std::move(key), interned_shapes_.size() + 1).first;
      shape = found->second;
    }
    token_shapes_.emplace(type.get(), shape);
    return shape;
  }

  /// A zero topology is a type with no completion facts, including data-only tuples.
  bool ContainsToken(const TypePtr& type) { return TokenShape(type) != 0; }

  /// Type expressions describe data geometry, not hidden completion-value storage.
  void VisitTypeMetadata(const TypePtr& type) {
    if (!type) return;
    const bool previous = in_metadata_;
    in_metadata_ = true;
    const auto [id, inserted] = RecordUse(type_metadata_uses_, type.get(), nullptr, true);
    if (!inserted) {
      in_metadata_ = previous;
      return;
    }
    const auto previous_parent = std::exchange(use_parent_, id);
    type_metadata::ForEachTypeMetadataChild(
        type, [this](const TypePtr& element) { VisitTypeMetadata(element); },
        [this](const ExprPtr& expression) { VisitExpressionUse(expression, true); },
        [this](const ExprPtr& descriptor) { VisitStorageMetadata(descriptor); });
    use_postorder_.push_back(id);
    use_parent_ = previous_parent;
    in_metadata_ = previous;
  }

  /// One shared field enumeration covers direct values and type back-references.
  /// Field roles affect other validators; every field is forbidden token storage here.
  void VisitStorageMetadata(const ExprPtr& descriptor) {
    if (!descriptor) return;
    const bool previous = in_metadata_;
    in_metadata_ = true;
    const auto [id, inserted] = RecordUse(storage_metadata_uses_, descriptor.get(), nullptr, true);
    if (!inserted) {
      in_metadata_ = previous;
      return;
    }
    const auto previous_parent = std::exchange(use_parent_, id);
    type_metadata::ForEachStorageMetadataExpr(
        descriptor, [this](const ExprPtr& expression, type_metadata::StorageMetadataField) {
          VisitExpressionUse(expression, true);
        });
    use_postorder_.push_back(id);
    use_parent_ = previous_parent;
    in_metadata_ = previous;
  }

  /// Tokens may only be created and used in one specialized outer device function.
  void CheckOuterContext(const Span& span, uint8_t contexts) {
    has_logical_sync_ = true;
    if ((function_->func_type_ != FunctionType::AIC && function_->func_type_ != FunctionType::AIV) ||
        HasContext(contexts, kBlockedContext)) {
      Error("Logical synchronization requires a specialized AIC/AIV function after inlining and outlining",
            span);
    }
    if (HasContext(contexts, kVFContext)) {
      Error("Outer-pipeline SyncToken operations and values cannot enter a RegScope", span);
    }
  }

  /// Re-run the registered schema rather than trusting a raw Call's stored result type.
  void RevalidateCall(const CallPtr& call) {
    try {
      auto inferred = OpRegistry::GetInstance()
                          .Create(call->op_->name_, call->args_, call->kwargs_, call->span_)
                          ->GetType();
      if (!structural_equal(inferred, call->GetType())) {
        Error("Logical synchronization call result type does not match its registered schema", call->span_);
      }
    } catch (const pypto::Error& error) {
      Error(error.what(), call->span_);
    }
  }

  /// Reject opaque call/container operations even when a malformed result hides its argument type.
  void RejectCallTokens(const std::vector<ExprPtr>& arguments, const TypePtr& result, const Span& span) {
    bool contains = ContainsToken(result);
    for (const auto& argument : arguments) contains = ContainsToken(argument->GetType()) || contains;
    if (contains) Error("SyncToken cannot pass through ordinary calls, Submit, or array operations", span);
  }

  /// Compiler metadata is not a value carrier for completion facts.
  void VisitMetadata(const std::vector<std::pair<std::string, std::any>>& metadata) {
    const bool previous = in_metadata_;
    in_metadata_ = true;
    for (const auto& [key, value] : metadata) {
      ForEachAttrExpr(value, [this](const ExprPtr& expression) { VisitExpressionUse(expression, true); });
    }
    in_metadata_ = previous;
  }

  /// Preserve token positions and source pipes through SSA without rescanning shared tuple subtrees.
  void CheckMatchingTypes(const TypePtr& first, const TypePtr& second, const Span& span) {
    if (TokenShape(first) != TokenShape(second)) {
      Error("SyncToken source-pipeline/type mismatch across assignment or control-flow boundary", span);
    }
  }

  /// Scalar predicates and loop bounds cannot reinterpret completion values as data.
  void RejectConditionToken(const ExprPtr& expression) {
    if (ContainsToken(expression->GetType())) {
      Error("SyncToken cannot be used as a control-flow condition or loop bound", expression->span_);
    }
  }

  /// Activate one token binding after its initializer; retain its first lifetime on redefinition.
  void Define(const VarPtr& variable) {
    VisitTypeMetadata(variable->GetType());
    if (!ContainsToken(variable->GetType())) return;
    CheckOuterContext(variable->span_, CurrentContexts());
    if (!lifetimes_.emplace(variable.get(), UseRange{++position_, kNoUse}).second) {
      Error("SyncToken value '" + variable->name_hint_ + "' has multiple SSA definitions", variable->span_);
      return;
    }
    introduced_.push_back(variable.get());
  }

  /// Remove only bindings introduced in a region, independent of outer scope size.
  void Restore(size_t checkpoint) {
    ++position_;
    while (introduced_.size() > checkpoint) {
      lifetimes_.at(introduced_.back()).last = position_;
      introduced_.pop_back();
    }
  }

  /// Report whether an SSA result vector contains completion values.
  template <typename Variable>
  bool AnyToken(const std::vector<Variable>& variables) {
    return std::any_of(variables.begin(), variables.end(),
                       [this](const auto& variable) { return ContainsToken(variable->GetType()); });
  }

  /// Find the region terminator by following only trailing SeqStmts edges.
  static YieldStmtPtr LastYield(StmtPtr body) {
    while (auto sequence = As<SeqStmts>(body)) {
      if (sequence->stmts_.empty()) return nullptr;
      body = sequence->stmts_.back();
    }
    return As<YieldStmt>(body);
  }

  /// Visit one region with isolated definitions and an explicit terminal-yield contract.
  void VisitBranch(const StmtPtr& body, const std::vector<VarPtr>& results) {
    const auto checkpoint = introduced_.size();
    const auto previous_yield = terminal_yield_;
    auto previous_results = std::move(yield_results_);
    auto terminal = LastYield(body);
    terminal_yield_ = terminal.get();
    yield_results_ = results;
    if (AnyToken(results) && !terminal) {
      Error("SyncToken control-flow results require a terminal YieldStmt", body->span_);
    }
    VisitStmt(body);
    Restore(checkpoint);
    terminal_yield_ = previous_yield;
    yield_results_ = std::move(previous_results);
  }

  /// Validate initial/carry/result types before entering a loop's strict lexical scope.
  void VisitLoop(const std::vector<IterArgPtr>& arguments, const std::vector<VarPtr>& results,
                 const StmtPtr& body, const ExprPtr& condition, const VarPtr& induction = nullptr) {
    for (const auto& argument : arguments) {
      VisitExpr(argument->initValue_);
      CheckMatchingTypes(argument->GetType(), argument->initValue_->GetType(), argument->span_);
    }
    const bool token_loop = AnyToken(arguments) || AnyToken(results);
    if (token_loop && arguments.size() != results.size()) {
      Error("SyncToken loop carry/result count mismatch", body->span_);
    }
    for (size_t index = 0; index < std::min(arguments.size(), results.size()); ++index) {
      CheckMatchingTypes(arguments[index]->GetType(), results[index]->GetType(), body->span_);
    }
    const auto checkpoint = introduced_.size();
    for (const auto& argument : arguments) Define(argument);
    if (induction) Define(induction);
    if (condition) {
      VisitExpr(condition);
      RejectConditionToken(condition);
    }
    std::vector<VarPtr> carries(arguments.begin(), arguments.end());
    VisitBranch(body, carries);
    Restore(checkpoint);
    for (const auto& result : results) Define(result);
  }

  /// Dispatch boundaries cannot inherit their surrounding function's token ownership.
  template <typename Scope>
  void VisitBlockedScope(const std::shared_ptr<const Scope>& statement) {
    const bool previous = blocked_;
    const auto checkpoint = introduced_.size();
    blocked_ = true;
    IRVisitor::VisitStmt_(statement);
    Restore(checkpoint);
    blocked_ = previous;
  }

  /// Collect recoverable verifier diagnostics rather than throwing halfway through the walk.
  void Error(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "LogicalSyncValid", 0, message, span);
  }

  FunctionPtr function_;
  std::vector<Diagnostic>& diagnostics_;
  // All graph/range state belongs to one function verification. Parent edges
  // are recorded only during first expansion; later roots add use sites only.
  std::unordered_map<const Expr*, size_t> expression_uses_;
  std::unordered_map<const Type*, size_t> type_metadata_uses_;
  std::unordered_map<const Expr*, size_t> storage_metadata_uses_;
  std::vector<UseRecord> uses_;
  std::vector<size_t> use_postorder_;
  std::unordered_map<const Var*, UseRange> lifetimes_;
  size_t use_parent_ = kNoUse;
  size_t position_ = 0;
  std::unordered_map<const Type*, size_t> token_shapes_;
  std::map<std::vector<size_t>, size_t> interned_shapes_;
  std::vector<const Var*> introduced_;
  std::vector<VarPtr> yield_results_;
  const YieldStmt* terminal_yield_ = nullptr;
  const Call* assignment_root_ = nullptr;
  const Call* evaluation_root_ = nullptr;
  ExprPtr nonlocal_call_;
  bool has_logical_sync_ = false;
  bool in_vf_ = false;
  bool blocked_ = false;
  bool in_metadata_ = false;
};

class LogicalSyncPropertyVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "LogicalSyncValid"; }

  /// Apply the function-local ownership contract independently to every invocation body.
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global, function] : program->functions_) VerifyLogicalSync(function, diagnostics);
  }
};

}  // namespace

void VerifyLogicalSync(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics) {
  if (function) LogicalSyncChecker(function, diagnostics).Run();
}

void CheckLogicalSync(const FunctionPtr& function) {
  std::vector<Diagnostic> diagnostics;
  VerifyLogicalSync(function, diagnostics);
  if (!diagnostics.empty()) {
    CHECK_SPAN(false, diagnostics.front().span) << diagnostics.front().message;
  }
}

PropertyVerifierPtr CreateLogicalSyncPropertyVerifier() {
  return std::make_shared<LogicalSyncPropertyVerifier>();
}

}  // namespace pypto::ir
