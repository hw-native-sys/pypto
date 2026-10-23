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
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

// Mat has no same-space copy instruction. A current/next prefetch therefore
// carries an allocation-slot selector, not the contents of a fixed L1 window.
// Keep that distinction in the IR before any planner sees it: a declared arena
// reserves all slots, while each Tile MemRef denotes its actual runtime slot.
// Codegen lowers slot selection to native handle selection so synchronization
// sees every possible backing bank, including loop-carried read-before-overwrite.
//
// Inventory and rewriting are fixed walks. The provenance walk claims each
// node once across candidates; intersecting candidates are conservatively left
// unchanged. Memoized seed/alias summaries keep the complete pass O(N).

ExprPtr Index(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

struct BranchResult {
  IfStmtPtr branch;
  size_t index;
  const ForStmt* loop;
};

class RotationInventory : public IRVisitor {
 public:
  std::unordered_map<const Var*, AssignStmtPtr> assignments;
  std::unordered_map<const Var*, const ForStmt*> owners;
  std::unordered_map<const Var*, const Stmt*> repetition_owners;
  std::unordered_map<const ForStmt*, const Stmt*> repetition_parents;
  std::unordered_map<const Var*, BranchResult> branches;
  std::unordered_map<const Var*, size_t> uses;
  std::unordered_map<const Var*, size_t> positions;
  std::vector<std::pair<VarPtr, size_t>> reads;
  std::unordered_set<std::string> names;
  std::vector<ForStmtPtr> loops;
  // Source -> destination binder for every region transfer. A rotating bank
  // must not escape into a second carry whose selector is not converted.
  std::vector<std::pair<VarPtr, const Var*>> boundary_uses;

  std::string Fresh(const std::string& stem) {
    std::string result;
    do {
      result = stem + "_rotation_" + std::to_string(next_name_++);
    } while (!names.insert(result).second);
    return result;
  }

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    ++uses[var.get()];
    names.insert(var->name_hint_);
    if (reading_ && As<TileType>(var->GetType())) reads.emplace_back(var, position_);
  }

  void VisitExpr_(const IterArgPtr& op) override {
    // Count the initializer only at the loop binder. The default visitor
    // recursively revisits it for every body use, falsely making the seed look
    // independently observed and expanding nested carry chains repeatedly.
    VisitVarLike_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    assignments.emplace(op->var_.get(), op);
    positions.emplace(op->var_.get(), ++position_);
    owners.emplace(op->var_.get(), loop_);
    repetition_owners.emplace(op->var_.get(), repetition_);
    names.insert(op->var_->name_hint_);
    VisitExpr(op->value_);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    ++position_;
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    const auto& registry = OpRegistry::GetInstance();
    const auto* entry =
        op->op_ && registry.IsRegistered(op->op_->name_) ? &registry.GetEntry(op->op_->name_) : nullptr;
    const bool outer_reading = reading_;
    for (size_t i = 0; i < op->args_.size(); ++i) {
      // Descriptor views and declarations do not keep the old contents live.
      // Unknown calls remain conservative readers; write-only destinations do
      // not count as old-value consumers.
      reading_ =
          !entry || (entry->GetExecutionMemoryAccessEvidence() != ExecutionMemoryAccessEvidence::NoAccess &&
                     !op_predicates::IsBufferAliasingViewOp(op->op_->name_) &&
                     (entry->GetIRStage() != OpIRStage::Functional ||
                      ArgEffectReads(entry->GetArgEffect(i, op->kwargs_))));
      VisitExpr(op->args_[i]);
    }
    reading_ = outer_reading;
  }

  void VisitExpr_(const SubmitPtr& op) override {
    const bool outer_reading = reading_;
    reading_ = true;
    IRVisitor::VisitExpr_(op);
    reading_ = outer_reading;
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      branches.emplace(op->return_vars_[i].get(), BranchResult{op, i, loop_});
      names.insert(op->return_vars_[i]->name_hint_);
    }
    VisitExpr(op->condition_);
    auto outer_targets = std::move(targets_);
    targets_ = op->return_vars_;
    VisitStmt(op->then_body_);
    if (op->else_body_) VisitStmt(*op->else_body_);
    targets_ = std::move(outer_targets);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    loops.push_back(op);
    repetition_parents.emplace(op.get(), repetition_);
    names.insert(op->loop_var_->name_hint_);
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    VisitInitializers(op->iter_args_, op->return_vars_);
    const auto* outer = loop_;
    const auto* outer_repetition = repetition_;
    loop_ = op.get();
    repetition_ = op.get();
    auto outer_targets = std::move(targets_);
    targets_ = op->return_vars_;
    VisitStmt(op->body_);
    targets_ = std::move(outer_targets);
    loop_ = outer;
    repetition_ = outer_repetition;
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    VisitInitializers(op->iter_args_, op->return_vars_);
    const auto* outer = loop_;
    const auto* outer_repetition = repetition_;
    loop_ = nullptr;  // A while body is not part of an enclosing For's rotation.
    repetition_ = op.get();
    auto outer_targets = std::move(targets_);
    targets_ = op->return_vars_;
    VisitExpr(op->condition_);
    VisitStmt(op->body_);
    targets_ = std::move(outer_targets);
    loop_ = outer;
    repetition_ = outer_repetition;
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    for (size_t i = 0; i < op->value_.size() && i < targets_.size(); ++i) {
      if (auto source = AsVarLike(op->value_[i])) boundary_uses.emplace_back(source, targets_[i].get());
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void VisitInitializers(const std::vector<IterArgPtr>& arguments, const std::vector<VarPtr>& results) {
    for (const auto& argument : arguments) {
      names.insert(argument->name_hint_);
      VisitExpr(argument->initValue_);
      if (auto source = AsVarLike(argument->initValue_)) {
        boundary_uses.emplace_back(source, argument.get());
      }
    }
    for (const auto& result : results) names.insert(result->name_hint_);
  }

  const ForStmt* loop_ = nullptr;
  const Stmt* repetition_ = nullptr;
  std::vector<VarPtr> targets_;
  size_t next_name_ = 0;
  size_t position_ = 0;
  bool reading_ = false;
};

struct Rotation {
  ForStmtPtr loop;
  size_t index = 0;
  bool valid = true;
  VarPtr seed;
  bool protect_seed = false;
  std::vector<VarPtr> values;
  std::vector<VarPtr> loads;
  std::unordered_map<const Var*, ExprPtr> tags;
  VarPtr base;
  IterArgPtr argument;
  VarPtr result;
  uint64_t slots = 0;
  int64_t seed_slot = 0;
};

struct SeedInfo {
  VarPtr root;
  bool observed_independently = false;
};

class RotationPlan {
 public:
  explicit RotationPlan(RotationInventory& inventory) : inventory(inventory) {}

  void Build() {
    for (const auto& loop : inventory.loops) {
      if (loop->kind_ != ForKind::Sequential) continue;
      auto yield = transform_utils::GetLastYieldStmt(loop->body_);
      if (!yield || yield->value_.size() != loop->iter_args_.size()) continue;
      for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
        auto type = As<TileType>(loop->iter_args_[i]->GetType());
        if (!IsUnboundMat(type)) continue;
        const auto initial = AsVarLike(loop->iter_args_[i]->initValue_);
        auto seed = FindSeed(initial);
        if (!seed.root || !SameDescriptor(type, seed.root->GetType())) continue;

        auto group = std::make_unique<Rotation>();
        group->loop = loop;
        group->index = i;
        group->seed = seed.root;
        // A seed outside the candidate's immediate repeating scope is read
        // again on the next entry, even when it has just one lexical use.
        // Preserve it across both enclosing for and while iterations.
        group->protect_seed =
            seed.observed_independently ||
            inventory.repetition_owners.at(seed.root.get()) != inventory.repetition_parents.at(loop.get());
        auto* rotation = group.get();
        groups.push_back(std::move(group));
        if (!Trace(loop->iter_args_[i], *rotation) || !Trace(AsVarLike(yield->value_[i]), *rotation) ||
            rotation->loads.empty()) {
          rotation->valid = false;
          continue;
        }
        auto [result_claim, unclaimed] = claims_.emplace(loop->return_vars_[i].get(), rotation);
        if (!unclaimed && result_claim->second != rotation) {
          result_claim->second->valid = false;
          rotation->valid = false;
        }
        // One seed cannot be rebound to two independently rotating arenas.
        // Preserve the input program rather than inventing a hidden Mat copy.
        auto [found, fresh] = seeds.emplace(seed.root.get(), rotation);
        if (!fresh) {
          found->second->valid = false;
          rotation->valid = false;
        }
      }
    }
    // A nested candidate's seed may be an outer candidate's producer. Such
    // intersecting ownership requires a joint rotation graph, not two arenas.
    for (const auto& [seed, group] : seeds) {
      auto claim = claims_.find(seed);
      if (claim != claims_.end() && claim->second != group) {
        claim->second->valid = false;
        group->valid = false;
      }
    }
    // A value exported through an unrelated phi/carry can remain live for an
    // arbitrary number of iterations. A finite private ring is not a valid
    // replacement unless that destination carries the same selector too.
    for (const auto& [source, target] : inventory.boundary_uses) {
      const auto* root = AliasRoot(source).get();
      auto owner = claims_.find(root);
      auto seed = seeds.find(root);
      auto* group = owner != claims_.end() ? owner->second : seed != seeds.end() ? seed->second : nullptr;
      if (!group || !group->valid) continue;
      // The protected initializer is outside the rotating ring. Other regions
      // may keep observing it without extending any rotating bank's lifetime.
      if (owner == claims_.end() && group->protect_seed) continue;
      auto destination = claims_.find(target);
      if ((destination != claims_.end() && destination->second == group) ||
          target == group->loop->return_vars_[group->index].get()) {
        continue;
      }
      group->valid = false;
    }
    std::unordered_map<const Var*, size_t> last_reads;
    for (const auto& [source, position] : inventory.reads) {
      auto& last = last_reads[AliasRoot(source).get()];
      last = std::max(last, position);
    }
    for (auto& group : groups) {
      if (!group->valid) continue;
      auto last = last_reads.find(group->loop->iter_args_[group->index].get());
      size_t first_load = inventory.positions.at(group->loads.front().get());
      for (const auto& load : group->loads) {
        first_load = std::min(first_load, inventory.positions.at(load.get()));
      }
      // Ordinary overwrite-only carries and load-after-consume loops already
      // fit one bank. Only reserve a ring when a fresh value is produced while
      // the old carry still has a genuine memory reader ahead of it. Lexical
      // ordering is conservative across mutually exclusive branches.
      if (last == last_reads.end() || last->second <= first_load) {
        group->valid = false;
        continue;
      }
      Prepare(*group);
      active_seeds.emplace(group->seed.get(), group.get());
      for (const auto& value : group->values) nodes.emplace(value.get(), group.get());
      nodes.emplace(group->loop->iter_args_[group->index].get(), group.get());
      nodes.emplace(group->loop->return_vars_[group->index].get(), group.get());
      loop_groups[group->loop.get()].emplace(group->index, group.get());
    }
  }

  ExprPtr Tag(Rotation& group, const VarPtr& value) {
    auto found = group.tags.find(value.get());
    if (found != group.tags.end()) return found->second;
    // Only bare aliases remain after leaf/phi tags have been registered.
    // Memoize the whole path so a long rename chain is traversed once.
    std::vector<const Var*> path;
    auto current = value;
    while (group.tags.count(current.get()) == 0) {
      path.push_back(current.get());
      const auto definition = inventory.assignments.find(current.get());
      INTERNAL_CHECK_SPAN(definition != inventory.assignments.end(), value->span_)
          << "Internal error: rotation alias has no definition";
      current = AsVarLike(definition->second->value_);
      INTERNAL_CHECK_SPAN(current, value->span_) << "Internal error: rotation alias has no source";
    }
    auto result = group.tags.at(current.get());
    for (const auto* alias : path) group.tags.emplace(alias, result);
    return result;
  }

  RotationInventory& inventory;
  std::vector<std::unique_ptr<Rotation>> groups;
  std::unordered_map<const Var*, Rotation*> nodes;
  std::unordered_map<const Var*, Rotation*> active_seeds;
  std::unordered_map<const ForStmt*, std::unordered_map<size_t, Rotation*>> loop_groups;

 private:
  static bool IsUnboundMat(const TileTypePtr& type) {
    if (!type || type->GetMemorySpace() != MemorySpace::Mat || type->memref_) return false;
    for (const auto& dimension : type->shape_) {
      auto constant = As<ConstInt>(dimension);
      if (!constant || constant->value_ <= 0) return false;
    }
    if (type->tile_view_) {
      for (const auto& extent : type->tile_view_->valid_shape) {
        if (!As<ConstInt>(extent)) return false;
      }
      if (type->tile_view_->start_offset) {
        auto offset = As<ConstInt>(type->tile_view_->start_offset);
        if (!offset || offset->value_ != 0) return false;
      }
    }
    return true;
  }

  bool SameDescriptor(const TileTypePtr& type, const TypePtr& other) {
    if (!IsUnboundMat(As<TileType>(other))) return false;
    return structural_equal(type, other);
  }

  SeedInfo FindSeed(const VarPtr& initial) {
    if (!initial) return {};
    std::vector<VarPtr> path;
    auto current = initial;
    SeedInfo result;
    while (current) {
      auto cached = seed_info_.find(current.get());
      if (cached != seed_info_.end()) {
        result = cached->second;
        break;
      }
      path.push_back(current);
      auto definition = inventory.assignments.find(current.get());
      if (definition == inventory.assignments.end()) break;
      if (auto call = As<Call>(definition->second->value_); IsOp(call, "tile.load")) {
        if (IsUnboundMat(As<TileType>(current->GetType()))) result.root = current;
        break;
      }
      auto alias = AsVarLike(definition->second->value_);
      if (!alias || !structural_equal(current->GetType(), alias->GetType())) break;
      current = alias;
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      result.observed_independently |= inventory.uses[(*it).get()] > 1;
      seed_info_.emplace(it->get(), result);
    }
    return result;
  }

  VarPtr AliasRoot(const VarPtr& value) {
    std::vector<VarPtr> path;
    auto current = value;
    while (alias_roots_.count(current.get()) == 0) {
      path.push_back(current);
      auto definition = inventory.assignments.find(current.get());
      auto source =
          definition == inventory.assignments.end() ? nullptr : AsVarLike(definition->second->value_);
      if (!source && definition != inventory.assignments.end()) {
        if (auto call = As<Call>(definition->second->value_); call && call->op_) {
          auto index = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
          if (!index && !call->args_.empty() && op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
            index = 0;
          }
          if (index) source = AsVarLike(call->args_[*index]);
        }
      }
      if (!source) break;
      current = source;
    }
    auto cached = alias_roots_.find(current.get());
    auto root = cached != alias_roots_.end() ? cached->second : current;
    for (const auto& alias : path) alias_roots_.emplace(alias.get(), root);
    return root;
  }

  bool Trace(const VarPtr& value, Rotation& group) {
    if (!value ||
        !SameDescriptor(As<TileType>(group.loop->iter_args_[group.index]->GetType()), value->GetType())) {
      return false;
    }
    auto [claim, fresh] = claims_.emplace(value.get(), &group);
    if (!fresh) {
      if (claim->second != &group) {
        claim->second->valid = false;
        return false;
      }
      return true;
    }
    group.values.push_back(value);
    if (value.get() == group.loop->iter_args_[group.index].get()) return true;
    auto assignment = inventory.assignments.find(value.get());
    if (assignment != inventory.assignments.end()) {
      if (inventory.owners.at(value.get()) != group.loop.get()) return false;
      if (auto source = AsVarLike(assignment->second->value_)) return Trace(source, group);
      if (auto call = As<Call>(assignment->second->value_); IsOp(call, "tile.load")) {
        group.loads.push_back(value);
        return true;
      }
      // Calls (including Submit) with arbitrary effects or alias contracts are
      // not fresh load leaves. Their storage must retain its existing lowering.
      return false;
    }
    auto branch = inventory.branches.find(value.get());
    if (branch == inventory.branches.end() || branch->second.loop != group.loop.get()) {
      return false;
    }
    const auto else_body = branch->second.branch->else_body_;
    if (!else_body) return false;
    auto then_yield = transform_utils::GetLastYieldStmt(branch->second.branch->then_body_);
    auto else_yield = transform_utils::GetLastYieldStmt(*else_body);
    const size_t index = branch->second.index;
    return then_yield && else_yield && index < then_yield->value_.size() &&
           index < else_yield->value_.size() && Trace(AsVarLike(then_yield->value_[index]), group) &&
           Trace(AsVarLike(else_yield->value_[index]), group);
  }

  void Prepare(Rotation& group) {
    const auto& argument = group.loop->iter_args_[group.index];
    const auto& result = group.loop->return_vars_[group.index];
    // K distinct producers may be simultaneously live with the current tile.
    // Give every producer its own displacement in a K+1-slot ring. An observed
    // initializer gets a separate immutable slot, outside that rotating ring.
    group.slots = group.loads.size() + 1 + static_cast<uint64_t>(group.protect_seed);
    group.seed_slot = group.protect_seed ? static_cast<int64_t>(group.slots - 1) : 0;
    group.base = std::make_shared<Var>(inventory.Fresh(argument->name_hint_ + "_banks"), GetPtrType(),
                                       argument->span_);
    group.argument = std::make_shared<IterArg>(inventory.Fresh(argument->name_hint_ + "_slot"),
                                               std::make_shared<ScalarType>(DataType::INDEX),
                                               Index(group.seed_slot, argument->span_), argument->span_);
    group.result = std::make_shared<Var>(inventory.Fresh(result->name_hint_ + "_slot"),
                                         std::make_shared<ScalarType>(DataType::INDEX), result->span_);
    group.tags.emplace(argument.get(), group.argument);
    group.tags.emplace(result.get(), group.result);
    for (const auto& load : group.loads) {
      group.tags.emplace(load.get(),
                         std::make_shared<Var>(inventory.Fresh(load->name_hint_ + "_slot"),
                                               std::make_shared<ScalarType>(DataType::INDEX), load->span_));
    }
    for (const auto& value : group.values) {
      if (inventory.branches.count(value.get()) != 0) {
        group.tags.emplace(
            value.get(), std::make_shared<Var>(inventory.Fresh(value->name_hint_ + "_slot"),
                                               std::make_shared<ScalarType>(DataType::INDEX), value->span_));
      }
    }
    for (const auto& value : group.values) Tag(group, value);
  }

  std::unordered_map<const Var*, Rotation*> claims_;
  std::unordered_map<const Var*, Rotation*> seeds;
  std::unordered_map<const Var*, SeedInfo> seed_info_;
  std::unordered_map<const Var*, VarPtr> alias_roots_;
};

class RotationRewriter : public IRMutator {
 public:
  explicit RotationRewriter(RotationPlan& plan) : plan_(plan) {
    for (const auto& group : plan_.groups) {
      if (!group->valid) continue;
      for (size_t i = 0; i < group->loads.size(); ++i) {
        load_ordinals_.emplace(group->loads[i].get(), i + 1);
      }
    }
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto seed = plan_.active_seeds.find(op->var_.get());
    auto node = plan_.nodes.find(op->var_.get());
    if (seed == plan_.active_seeds.end() && node == plan_.nodes.end()) {
      return IRMutator::VisitStmt_(op);
    }
    auto& group = *(seed != plan_.active_seeds.end() ? seed->second : node->second);
    ExprPtr tag =
        seed != plan_.active_seeds.end() ? Index(group.seed_slot, op->span_) : plan_.Tag(group, op->var_);
    auto var = SlottedVar(op->var_, group, tag);
    var_remap_[op->var_.get()] = var;
    auto result = MutableCopy(op);
    result->var_ = var;
    result->value_ = VisitExpr(op->value_);
    auto ordinal = load_ordinals_.find(op->var_.get());
    if (ordinal == load_ordinals_.end()) return result;

    const auto rotating_slots = static_cast<int64_t>(group.loads.size() + 1);
    auto next = MakeFloorMod(
        MakeAdd(group.argument, Index(static_cast<int64_t>(ordinal->second), op->span_), op->span_),
        Index(rotating_slots, op->span_), op->span_);
    auto selector = AsVarLike(tag);
    INTERNAL_CHECK_SPAN(selector, op->span_) << "Internal error: load selector must be an SSA variable";
    return SeqStmts::Flatten({std::make_shared<AssignStmt>(selector, next, op->span_), result}, op->span_);
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    auto result = MutableCopy(op);
    for (size_t i = 0; i < result->value_.size(); ++i) {
      auto group = yield_groups_.find(i);
      result->value_[i] = group == yield_groups_.end() ? VisitExpr(op->value_[i])
                                                       : plan_.Tag(*group->second, AsVarLike(op->value_[i]));
    }
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = MutableCopy(op);
    result->condition_ = VisitExpr(op->condition_);
    auto outer = std::move(yield_groups_);
    yield_groups_.clear();
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      auto group = plan_.nodes.find(op->return_vars_[i].get());
      if (group == plan_.nodes.end()) continue;
      yield_groups_.emplace(i, group->second);
      result->return_vars_[i] = AsVarLike(plan_.Tag(*group->second, op->return_vars_[i]));
    }
    result->then_body_ = VisitStmt(op->then_body_);
    if (op->else_body_) result->else_body_ = VisitStmt(*op->else_body_);
    std::vector<StmtPtr> statements{result};
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      auto group = yield_groups_.find(i);
      if (group == yield_groups_.end()) continue;
      statements.push_back(View(op->return_vars_[i], *group->second, result->return_vars_[i]));
    }
    yield_groups_ = std::move(outer);
    return SeqStmts::Flatten(std::move(statements), op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto outer = std::move(yield_groups_);
    yield_groups_.clear();
    auto groups = plan_.loop_groups.find(op.get());
    if (groups == plan_.loop_groups.end()) {
      auto result = IRMutator::VisitStmt_(op);
      yield_groups_ = std::move(outer);
      return result;
    }
    yield_groups_ = groups->second;
    auto result = MutableCopy(op);
    result->start_ = VisitExpr(op->start_);
    result->stop_ = VisitExpr(op->stop_);
    result->step_ = VisitExpr(op->step_);
    result->attrs_ = MutateScopeAttrs(op->attrs_).first;
    std::vector<StmtPtr> prefix;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto group = groups->second.find(i);
      if (group != groups->second.end()) {
        result->iter_args_[i] = group->second->argument;
        result->return_vars_[i] = group->second->result;
        prefix.push_back(View(op->iter_args_[i], *group->second, group->second->argument));
      } else {
        auto argument = As<IterArg>(IRMutator::VisitExpr_(op->iter_args_[i]));
        result->iter_args_[i] = argument;
        var_remap_[op->iter_args_[i].get()] = argument;
        result->return_vars_[i] = AsVarLike(VisitExpr(op->return_vars_[i]));
      }
    }
    prefix.push_back(VisitStmt(op->body_));
    result->body_ = SeqStmts::Flatten(std::move(prefix), op->span_);
    for (const auto& argument : op->iter_args_) var_remap_.erase(argument.get());
    std::vector<StmtPtr> statements{result};
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      auto group = groups->second.find(i);
      if (group != groups->second.end()) {
        statements.push_back(View(op->return_vars_[i], *group->second, group->second->result));
      }
    }
    yield_groups_ = std::move(outer);
    return SeqStmts::Flatten(std::move(statements), op->span_);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto outer = std::move(yield_groups_);
    yield_groups_.clear();
    auto result = IRMutator::VisitStmt_(op);
    yield_groups_ = std::move(outer);
    return result;
  }

 private:
  VarPtr SlottedVar(const VarPtr& original, const Rotation& group, const ExprPtr& tag) {
    auto memref = std::make_shared<MemRef>(group.base, int64_t{0}, uint64_t{0}, original->span_,
                                           /*is_pinned=*/true, group.slots, tag);
    auto type = CloneTypeWithMemRef(original->GetType(), memref);
    return std::make_shared<Var>(original->name_hint_, type, original->span_);
  }

  StmtPtr View(const VarPtr& original, const Rotation& group, const ExprPtr& tag) {
    auto var = SlottedVar(original, group, tag);
    var_remap_[original.get()] = var;
    auto tile = As<TileType>(var->GetType());
    auto shape = std::make_shared<MakeTuple>(tile->shape_, original->span_);
    auto call = OpRegistry::GetInstance().Create(
        "tile.create", {shape}, {{"dtype", tile->dtype_}, {"target_memory", MemorySpace::Mat}},
        original->span_);
    return std::make_shared<AssignStmt>(var, call, original->span_);
  }

  RotationPlan& plan_;
  std::unordered_map<size_t, Rotation*> yield_groups_;
  std::unordered_map<const Var*, size_t> load_ordinals_;
};

FunctionPtr TransformLowerMatValueRotation(const FunctionPtr& function) {
  if (IsOrchestrationLike(function) || function->ir_stage_ == FunctionIRStage::Buffer) return function;
  const auto* context = PassContext::Current();
  // Addr-less multi-buffer planning has separate synchronization requirements.
  // Preserve that existing path until it can express the same explicit banks.
  if (context && context->GetMemoryPlanner() == MemoryPlanner::PtoAS) return function;
  RotationInventory inventory;
  for (const auto& parameter : function->params_) inventory.names.insert(parameter->name_hint_);
  inventory.VisitStmt(function->body_);
  RotationPlan plan(inventory);
  plan.Build();
  if (plan.nodes.empty()) return function;
  auto body = RotationRewriter(plan).VisitStmt(function->body_);
  auto result = MutableCopy(function);
  result->body_ = body;
  return result;
}

}  // namespace

namespace pass {
Pass LowerMatValueRotation() {
  return CreateFunctionPass(TransformLowerMatValueRotation, "LowerMatValueRotation",
                            kLowerMatValueRotationProperties);
}
}  // namespace pass
}  // namespace pypto::ir
