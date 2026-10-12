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

#include "pypto/ir/transforms/utils/software_pipeline.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/io_category.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/nested_pipeline.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/pipeline_loop_utils.h"
#include "pypto/ir/transforms/utils/tile_buf_signature.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

using pipeline_loop::MakeConstIndex;
using pipeline_loop::OffsetIndex;
using pipeline_loop::SplitBodyYield;

/// A fixed number of linear walks plus indexed lookups: O(N log N). Stage count
/// is bounded to 2..4, so cloning the short prologue / epilogue is constant-factor.
class ExprReads : public IRVisitor {
 public:
  std::set<const Var*> vars;
  bool has_call = false;

  void VisitVarLike_(const VarPtr& var) override { vars.insert(var.get()); }
  void VisitExpr_(const CallPtr& call) override {
    has_call = true;
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    has_call = true;
    IRVisitor::VisitExpr_(submit);
  }
};

bool IsDenseVecTile(const TileTypePtr& type) {
  if (!type || type->memory_space_ != MemorySpace::Vec || type->memref_.has_value() ||
      type->shape_.size() != 2 || type->dtype_.GetBit() < 8) {
    return false;
  }
  for (const auto& dim : type->shape_) {
    const auto value = transform_utils::EvalConstInt(dim);
    if (!value || *value <= 0) return false;
  }
  if (!type->tile_view_) return true;
  const auto& view = *type->tile_view_;
  if (view.slayout != TileLayout::none_box) return false;
  if (view.blayout != TileLayout::row_major &&
      !(view.blayout == TileLayout::col_major && transform_utils::EvalConstInt(type->shape_[1]) == 1)) {
    return false;
  }
  if (view.start_offset && transform_utils::EvalConstInt(view.start_offset) != 0) return false;
  if (!view.valid_shape.empty()) {
    if (view.valid_shape.size() != type->shape_.size()) return false;
    for (size_t i = 0; i < view.valid_shape.size(); ++i) {
      if (transform_utils::EvalConstInt(view.valid_shape[i]) !=
          transform_utils::EvalConstInt(type->shape_[i])) {
        return false;
      }
    }
  }
  return true;
}

struct LoopPlan {
  NestedPipelinePlan nesting;
  std::map<const Var*, const Var*> nested_owners;
  std::map<const Var*, NestedPipelineSlot> slot_axes;
  std::map<const Var*, CallPtr> scope_definitions;
  std::set<const Var*> local_workspace;
  std::set<const Var*> prefetched_loads;
  std::set<const Var*> head_loads;
  // Anchor -> parent-iteration distance -> producer slice. Distances are at
  // most S-1, including when a child has fewer iterations than its depth.
  std::map<const Var*, std::map<int64_t, std::set<const Var*>>> prefetch_issues;
  std::map<int64_t, std::set<const Var*>> prologue_loads;
  bool nested_storage_valid = true;
  int64_t slots = 0;
  int64_t trips = 0;
  bool dynamic_trips = false;
  bool needs_overflow_fallback = false;
  int64_t start = 0;
  int64_t step = 0;
  std::set<const Var*> loads;
  std::set<const Var*> fifo_inputs;
  std::set<const Var*> partial_aliases;
  std::map<const Var*, int64_t> region_slots;
  int fifo_slots = 0;
  std::set<const Var*> scalars;
  std::set<const Var*> store_sources;
  std::map<const Var*, const Var*> aliases;
  /// Slotted tile -> the input load owning its region.
  std::map<const Var*, const Var*> slot_roots;
  std::map<const Var*, VarPtr> bases;
  std::vector<ExprPtr> yields;
};

/// A whole-tile view may change dtype at equal element width or reshape a
/// one-dimensional carrier. Reuse the backend signature's layout proof.
bool CompatibleSlotTypes(const TypePtr& lhs, const TypePtr& rhs) {
  auto a = As<TileType>(lhs);
  auto b = As<TileType>(rhs);
  if (!a || !b || a->dtype_.GetBit() != b->dtype_.GetBit()) return false;
  auto sa = TileBufSignature::FromTileType(*a);
  auto sb = TileBufSignature::FromTileType(*b);
  return sa.IsFullSlotAlias(sb);
}

class TileDefinitions : public IRVisitor {
 public:
  std::map<const Var*, CallPtr> calls;
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (As<TileType>(stmt->var_->GetType())) calls.emplace(stmt->var_.get(), As<Call>(stmt->value_));
    IRVisitor::VisitStmt_(stmt);
  }
};

/// Eligibility deliberately requires direct ordinary GM parameters. Tensor views,
/// distributed windows and aliases outside the loop need a broader alias proof;
/// rejecting them keeps prefetch from crossing a hidden read-after-write edge.
class SoftwarePipelineMutator : public IRMutator {
 public:
  explicit SoftwarePipelineMutator(const FunctionPtr& func, const ForStmt* fifo_loop = nullptr,
                                   int fifo_slots = 0)
      : io_(IOCategoryOps::Build()), fifo_loop_(fifo_loop), fifo_slots_(fifo_slots) {
    for (const auto& param : func->params_) {
      if (As<TensorType>(param->GetType())) tensor_params_.insert(param.get());
    }
    definitions_.VisitStmt(func->body_);
  }
  bool changed = false;

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (auto scalar = As<ScalarType>(op->var_->GetType());
        scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX)) {
      scalar_bounds_.const_int_bound.Update(op->var_, scalar_bounds_.const_int_bound(op->value_));
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ == ForKind::Unroll) return op;
    if (fifo_loop_ && op.get() != fifo_loop_) return IRMutator::VisitStmt_(op);
    if (op->kind_ != ForKind::Pipeline) return IRMutator::VisitStmt_(op);
    // A declined parent will be replicated. Never synthesize a pinned region
    // below it, which would then acquire several co-live users of the same slot.
    if (inside_pipeline_) return op;
    LoopPlan plan;
    std::string reason;
    bool prepared = PrepareNestedPipeline(op, plan.nesting, reason, fifo_loop_ != nullptr);
    if (prepared && plan.nesting.nested) {
      TileDefinitions scoped;
      scoped.VisitStmt(plan.nesting.loop->body_);
      plan.scope_definitions = std::move(scoped.calls);
    }
    if (!prepared || !Analyze(plan.nesting.loop, plan, reason)) {
      LOG_DEBUG << "LowerPipelineToSlots: software pipeline declined for '" << op->loop_var_->name_hint_
                << "' at " << op->span_.to_string() << ": " << reason;
      inside_pipeline_ = true;
      auto result = IRMutator::VisitStmt_(op);
      inside_pipeline_ = false;
      return result;
    }
    changed = true;
    if (plan.nesting.nested || plan.fifo_slots) return BuildScoped(plan.nesting.loop, plan);
    return Build(op, plan);
  }

 private:
  static bool Decline(std::string& reason, const char* message) {
    reason = message;
    return false;
  }

  static bool CollectAssignments(const StmtPtr& stmt, bool fifo, int depth,
                                 std::vector<AssignStmtPtr>& assigns, std::set<const Var*>& guarded,
                                 std::string& reason) {
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) {
        if (!CollectAssignments(child, fifo, depth, assigns, guarded, reason)) return false;
      }
      return true;
    }
    if (auto assign = As<AssignStmt>(stmt)) {
      assigns.push_back(assign);
      if (depth) guarded.insert(assign->var_.get());
      return true;
    }
    if (fifo) {
      if (auto branch = As<IfStmt>(stmt)) {
        ExprReads reads;
        reads.VisitExpr(branch->condition_);
        if (reads.has_call || !branch->return_vars_.empty() || branch->else_body_) {
          return Decline(reason, "FIFO guards require scalar conditions and no branch results/else");
        }
        for (const auto* var : reads.vars) {
          if (!As<ScalarType>(var->GetType())) return Decline(reason, "guard depends on tile data");
        }
        return CollectAssignments(branch->then_body_, fifo, depth + 1, assigns, guarded, reason);
      }
      if (auto eval = As<EvalStmt>(stmt);
          eval && depth == 0 && IsOp(As<Call>(eval->expr_), "system.tfree_to_aic")) {
        return true;
      }
    }
    return Decline(reason, "control flow, task launches or side-effect statements are unsupported");
  }

  bool AnalyzeScalar(const AssignStmtPtr& assign, LoopPlan& plan, std::set<const Var*>& tensors,
                     std::string& reason) const {
    ExprReads reads;
    reads.VisitExpr(assign->value_);
    if (reads.has_call) {
      auto call = As<Call>(assign->value_);
      if ((!plan.fifo_slots && !plan.nesting.nested) || !call ||
          !OpRegistry::GetInstance().IsRegistered(call->op_->name_)) {
        return Decline(reason, "scalar calls cannot be duplicated into prefetch");
      }
      const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
      if (entry.IsNoDuplicate() || entry.GetCrossCoreRole() ||
          (!entry.HasDeclaredArgEffects() &&
           entry.GetExecutionMemoryAccessEvidence() != ExecutionMemoryAccessEvidence::NoAccess)) {
        return Decline(reason, "scalar call lacks a readonly effect contract");
      }
      for (size_t i = 0; i < call->args_.size(); ++i) {
        ExprReads nested;
        nested.VisitExpr(call->args_[i]);
        if (nested.has_call || entry.GetArgEffect(i, call->kwargs_) != ArgEffect::Read) {
          return Decline(reason, "scalar call has nested calls or writes");
        }
      }
    }
    for (const auto* var : reads.vars) {
      if ((plan.fifo_slots || plan.nesting.nested) && tensor_params_.count(var)) {
        tensors.insert(var);
      } else if (!As<ScalarType>(var->GetType()) || var->GetKind() == ObjectKind::IterArg) {
        return Decline(reason, "scalar address computation depends on carried or tile data");
      }
    }
    plan.scalars.insert(assign->var_.get());
    return true;
  }

  bool Analyze(const ForStmtPtr& op, LoopPlan& plan, std::string& reason) {
    plan.slots = op->GetAttr<int>(kPipelineStagesAttr, 0);
    plan.fifo_slots = fifo_loop_ ? fifo_slots_ : 0;
    auto start = transform_utils::EvalConstInt(op->start_);
    auto stop = transform_utils::EvalConstInt(op->stop_);
    auto step = transform_utils::EvalConstInt(op->step_);
    if (plan.slots < 2 || plan.slots > 4 || !start || !step || *start < 0 || *step <= 0) {
      return Decline(reason, "requires stage 2..4, static nonnegative start and positive step");
    }
    plan.start = *start;
    plan.step = *step;
    plan.dynamic_trips = !stop.has_value();
    if (plan.nesting.nested && !PassContext::Current()->GetBackendHandler()->RequiresGMPipeBuffer()) {
      return Decline(reason, "backend does not support nested slot event boundaries");
    }
    if (plan.dynamic_trips && !PassContext::Current()->GetBackendHandler()->RequiresGMPipeBuffer()) {
      return Decline(reason, "backend does not support runtime-bound slot events");
    }
    const auto max_stop = std::max(plan.start, scalar_bounds_.const_int_bound(op->stop_).max_value);
    const auto max_extent = max_stop - plan.start;
    const auto max_trips = max_extent / plan.step + (max_extent % plan.step != 0);
    if (plan.nesting.nested &&
        (plan.start != 0 || plan.step != 1 ||
         max_trips >
             (std::numeric_limits<int64_t>::max() - 4 * plan.nesting.max_stride) / plan.nesting.max_stride)) {
      return Decline(reason, "nested pipeline requires bounded normalized outer ordinals");
    }
    plan.needs_overflow_fallback =
        plan.dynamic_trips && max_trips > std::numeric_limits<int64_t>::max() - plan.slots;
    if (stop) plan.trips = transform_utils::ComputeStaticTripCount(*start, *stop, *step);
    ExprReads bound_reads;
    bound_reads.VisitExpr(op->stop_);
    if (bound_reads.has_call) return Decline(reason, "loop bound contains a call or task launch");
    if (plan.fifo_slots &&
        (plan.start != 0 || plan.step != 1 || plan.needs_overflow_fallback || !op->iter_args_.empty())) {
      return Decline(reason, "FIFO plan requires bounded unit ordinals without carried data");
    }
    if (!plan.fifo_slots && !plan.nesting.nested && !plan.dynamic_trips && plan.trips < plan.slots - 1) {
      return Decline(reason, "trip count is smaller than prefetch distance");
    }
    auto [body, yields] = SplitBodyYield(op->body_);
    plan.yields = std::move(yields);
    auto seq = As<SeqStmts>(body);
    if (!seq || plan.yields.size() != op->iter_args_.size()) {
      return Decline(reason, "requires a straight-line body with matching yields");
    }
    std::vector<AssignStmtPtr> statements;
    std::set<const Var*> guarded;
    if (!CollectAssignments(body, plan.fifo_slots != 0 || plan.nesting.nested, 0, statements, guarded,
                            reason)) {
      return false;
    }
    std::map<const Var*, const Var*> tensor_roots;
    for (const auto& iter_arg : op->iter_args_) {
      auto init = AsVarLike(iter_arg->initValue_);
      // Exact TensorType intentionally excludes distributed window carries.
      if (!As<TensorType>(iter_arg->GetType()) || !init || !tensor_params_.count(init.get())) {
        return Decline(reason, "only ordinary output-tensor pass-through carries are supported");
      }
      tensor_roots[iter_arg.get()] = init.get();
    }
    std::set<const Var*> read_tensors;
    std::set<const Var*> written_tensors;
    std::set<const Var*> local_tiles;
    std::map<const Var*, size_t> consumer_counts;
    std::set<const Var*> non_store_consumers;
    std::set<const Var*> scratch_tiles;
    std::set<const Var*> read_tiles;
    for (const auto& assign : statements) {
      ExprReads reads;
      reads.VisitExpr(assign->value_);
      const auto consumer = As<Call>(assign->value_);
      const bool is_store = consumer && io_.IsStoreLike(consumer->op_);
      const bool is_view =
          consumer && op_predicates::IsBufferAliasingViewOp(consumer->op_->name_) &&
          OpRegistry::GetInstance().GetEntry(consumer->op_->name_).GetExecutionMemoryAccessEvidence() ==
              ExecutionMemoryAccessEvidence::NoAccess;
      if (is_view && !consumer->args_.empty()) {
        auto source = AsVarLike(consumer->args_[0]);
        if (!source) return Decline(reason, "view has no source variable");
        if (!CompatibleSlotTypes(assign->var_->GetType(), source->GetType())) {
          if (!plan.fifo_slots && !plan.nesting.nested) {
            return Decline(reason, "view does not preserve a complete slot's physical storage");
          }
          plan.partial_aliases.insert(assign->var_.get());
        }
        plan.aliases[assign->var_.get()] = AliasRoot(plan, source.get());
      }
      // ExprReads deduplicates operands within one statement: add(x, x) is one
      // consumer, while two separate computations of x are two consumers.
      for (const auto* var : reads.vars) {
        const auto* root = AliasRoot(plan, var);
        if (As<TileType>(var->GetType()) && !is_view) {
          ++consumer_counts[root];
          if (!is_store) non_store_consumers.insert(root);
        }
      }
      if (As<ScalarType>(assign->var_->GetType())) {
        if (!AnalyzeScalar(assign, plan, read_tensors, reason)) return false;
      } else {
        auto call = As<Call>(assign->value_);
        if (!call || !call->op_ || !OpRegistry::GetInstance().IsRegistered(call->op_->name_)) {
          return Decline(reason, "requires registered tile operations; submits and aliases are unsupported");
        }
        for (const auto& arg : call->args_) {
          ExprReads arg_reads;
          arg_reads.VisitExpr(arg);
          if (arg_reads.has_call) return Decline(reason, "nested calls are unsupported");
        }
        if (plan.fifo_slots && IsOp(call, "tile.tpop_from_aic")) {
          if (guarded.count(assign->var_.get()) || !IsDenseVecTile(As<TileType>(assign->var_->GetType()))) {
            return Decline(reason, "FIFO receive must be unconditional and full-box");
          }
          plan.fifo_inputs.insert(assign->var_.get());
          plan.loads.insert(assign->var_.get());
          DeclareRegion(plan, assign->var_);
          local_tiles.insert(assign->var_.get());
        } else if (io_.IsLoadLike(call->op_)) {
          if (!AnalyzeLoad(assign, call, plan, read_tensors, reason)) return false;
          local_tiles.insert(assign->var_.get());
        } else if (io_.IsStoreLike(call->op_)) {
          if (!AnalyzeStore(assign, call, tensor_params_, tensor_roots, written_tensors, reason)) {
            return false;
          }
          const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
          for (size_t i = 0; i < call->args_.size(); ++i) {
            if (!As<TileType>(call->args_[i]->GetType()) ||
                !ArgEffectReads(entry.GetArgEffect(i, call->kwargs_))) {
              continue;
            }
            auto source = AsVarLike(call->args_[i]);
            if (!source || !local_tiles.count(source.get())) {
              return Decline(reason, "store source is not a local tile with known storage ownership");
            }
            plan.store_sources.insert(source.get());
          }
        } else {
          if (!AnalyzeCompute(assign, call, local_tiles, plan, scratch_tiles, read_tiles, reason)) {
            return false;
          }
          local_tiles.insert(assign->var_.get());
        }
      }
    }
    if ((plan.loads.empty() && plan.fifo_inputs.empty()) || written_tensors.empty()) {
      return Decline(reason, "requires both a GM load and store");
    }
    for (const auto* scratch : scratch_tiles) {
      if (read_tiles.count(scratch)) return Decline(reason, "workspace is also read as loop data");
    }
    for (const auto* scratch : plan.local_workspace) {
      if (!scratch_tiles.count(scratch) || plan.store_sources.count(scratch)) {
        return Decline(reason, "nested tile.create must be used exclusively as workspace");
      }
    }
    for (const auto* root : read_tensors) {
      if (written_tensors.count(root)) {
        return Decline(reason, "a prefetched input may alias a written tensor");
      }
    }
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto value = AsVarLike(plan.yields[i]);
      if (!value || !tensor_roots.count(value.get()) ||
          tensor_roots.at(value.get()) != tensor_roots.at(op->iter_args_[i].get())) {
        return Decline(reason, "yield changes tensor identity or carries computed data");
      }
    }
    for (const auto* load : plan.loads) {
      if (consumer_counts[load] > 1 && !plan.nesting.nested) {
        return Decline(reason, "a prefetched tile has multiple consumer statements");
      }
    }
    for (const auto* source : plan.store_sources) {
      if (non_store_consumers.count(AliasRoot(plan, source))) {
        return Decline(reason, "a stored tile is also consumed by compute");
      }
    }
    if (plan.fifo_slots && plan.fifo_inputs.size() != 1) {
      return Decline(reason, "requires exactly one FIFO input");
    }
    // Only live iteration versions need slots here. Output allocation, in-place
    // reuse and the final capacity check belong to the memory-planning passes.
    if (!plan.nested_storage_valid) {
      return Decline(reason, "nested storage changes shape, stride or depth across specializations");
    }
    for (auto* alias : plan.partial_aliases) {
      auto* root = AliasRoot(plan, alias);
      if (plan.slot_roots.count(root) || plan.fifo_inputs.count(root)) {
        return Decline(reason, "partial views of rotating or borrowed input storage are unsupported");
      }
    }
    if ((plan.nesting.nested || plan.fifo_slots) && !PlanProducerIssues(plan, reason)) return false;
    // Fixed-address multi_tile buffers require their physical slot stride to
    // satisfy target alignment. This is an encoding check, not a UB budget.
    const auto* handler = PassContext::Current()->GetBackendHandler();
    for (const auto& [owner, base] : plan.bases) {
      auto bytes =
          utils::StaticPhysicalAllocationBytes(As<TileType>(owner->GetType()), MemorySpace::Vec, handler);
      if (!bytes) return Decline(reason, "slot size is not statically known");
      if (backend::BackendConfig::IsConfigured()) {
        const auto alignment = backend::GetBackend()->GetMemAlignment(MemorySpace::Vec);
        if (alignment > 0 && *bytes % alignment != 0) {
          return Decline(reason, "slot stride is not aligned for a multi-buffer region");
        }
      }
    }
    return true;
  }

  static bool PlanProducerIssues(LoopPlan& plan, std::string& reason) {
    using Guards = std::vector<const IfStmt*>;
    std::map<const Var*, Guards> guards;
    Guards path;
    std::function<void(const StmtPtr&)> collect = [&](const StmtPtr& stmt) {
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) collect(child);
      } else if (auto branch = As<IfStmt>(stmt)) {
        path.push_back(branch.get());
        collect(branch->then_body_);
        path.pop_back();
      } else if (auto assign = As<AssignStmt>(stmt);
                 assign && plan.nesting.anchors.count(assign->var_.get())) {
        guards.emplace(assign->var_.get(), path);
      }
    };
    collect(plan.nesting.loop->body_);
    std::map<std::pair<const Var*, int64_t>, const Var*> members;
    std::map<const Var*, bool> continuous;
    std::map<const Var*, size_t> family_size;
    for (const auto* load : plan.loads) {
      auto found = plan.nesting.storage.find(load);
      if (found == plan.nesting.storage.end()) {
        plan.prefetched_loads.insert(load);
        const auto distance = plan.fifo_inputs.count(load)
                                  ? std::min<int64_t>(plan.slots, plan.fifo_slots) - 1
                                  : plan.region_slots.at(plan.slot_roots.at(load)) - 1;
        plan.prefetch_issues[nullptr][distance].insert(load);
        for (int64_t initial = 0; initial < distance; ++initial) plan.prologue_loads[initial].insert(load);
        continue;
      }
      const auto& axis = found->second;
      if (!members.emplace(std::make_pair(axis.family, axis.phase), load).second) {
        return Decline(reason, "nested stream has ambiguous phase ownership");
      }
      continuous.try_emplace(axis.family, true);
      continuous[axis.family] &= guards.at(axis.anchor).empty();
      ++family_size[axis.family];
    }
    // A short child can leave most of its stage slots unused. Limit continuous
    // lookahead by the actual distance between uses of each hierarchical slot:
    // parent=2, child stage=4, child trips=1 must not preload three parents into
    // two parent contexts. This changes issue distance, never allocation depth.
    // Phase specialization is bounded; indexed family/residue walks are O(N log N).
    std::map<const Var*, int64_t> reuse_distance;
    std::map<const Var*, int64_t> family_period;
    std::map<const Var*, std::map<int64_t, std::pair<int64_t, int64_t>>> phase_spans;
    for (const auto& [member, load] : members) {
      const auto& axis = plan.nesting.storage.at(load);
      const auto period = plan.slots * axis.stride;
      family_period.emplace(axis.family, period);
      auto gap = reuse_distance.try_emplace(axis.family, period).first;
      auto& spans = phase_spans[axis.family];
      auto previous = spans.find(axis.storage_phase);
      if (previous == spans.end()) {
        spans.emplace(axis.storage_phase, std::make_pair(axis.phase, axis.phase));
      } else {
        gap->second = std::min(gap->second, axis.phase - previous->second.second);
        previous->second.second = axis.phase;
      }
    }
    for (const auto& [family, spans] : phase_spans) {
      for (const auto& [residue, span] : spans) {
        reuse_distance[family] =
            std::min(reuse_distance.at(family), family_period.at(family) + span.first - span.second);
      }
    }
    for (const auto* load : plan.loads) {
      auto found = plan.nesting.storage.find(load);
      if (found == plan.nesting.storage.end()) continue;
      const auto& axis = found->second;
      if (continuous.at(axis.family) && family_size.at(axis.family) == static_cast<size_t>(axis.stride)) {
        const auto distance = std::min(axis.slots - 1, reuse_distance.at(axis.family) - 1);
        const auto issue_phase = ((axis.phase - distance) % axis.stride + axis.stride) % axis.stride;
        const auto parent_distance = (issue_phase + distance - axis.phase) / axis.stride;
        auto earlier = members.find({axis.family, issue_phase});
        INTERNAL_CHECK(earlier != members.end()) << "Missing phase in a complete nested stream";
        // The first phase has no earlier consumer in this parent iteration.
        // Its producer slice can overlap parent compute at the loop head.
        auto anchor = issue_phase ? plan.nesting.storage.at(earlier->second).anchor : nullptr;
        plan.prefetch_issues[anchor][parent_distance].insert(load);
        plan.prefetched_loads.insert(load);
        for (int64_t ordinal = axis.phase; ordinal < distance; ordinal += axis.stride) {
          plan.prologue_loads[ordinal / axis.stride].insert(load);
        }
        continue;
      }
      // Predicated ancestor scopes retain their local staging window. Moving
      // a producer into a different dynamic invocation would require proving
      // that the destination anchor executes whenever that producer is needed.
      const auto window = std::min(axis.slots, reuse_distance.at(axis.family));
      if (axis.phase < window) {
        plan.prefetched_loads.insert(load);
        plan.head_loads.insert(load);
      }
      if (axis.phase < window) continue;
      auto earlier = members.find({axis.family, axis.phase - window + 1});
      if (earlier == members.end()) continue;
      auto anchor = plan.nesting.storage.at(earlier->second).anchor;
      // Do not issue under an earlier enclosing predicate that can be false
      // while the future consumer's enclosing predicate is true. Such loads
      // remain at their original phase; the enclosing plan stays valid.
      if (guards.at(anchor) != guards.at(axis.anchor)) continue;
      plan.prefetch_issues[anchor][0].insert(load);
      plan.prefetched_loads.insert(load);
    }
    return true;
  }

  static const Var* AliasRoot(const LoopPlan& plan, const Var* var) {
    auto it = plan.aliases.find(var);
    return it == plan.aliases.end() ? var : it->second;
  }

  bool AnalyzeLoad(const AssignStmtPtr& assign, const CallPtr& call, LoopPlan& plan,
                   std::set<const Var*>& reads, std::string& reason) {
    if (!IsOp(call, "tile.load") || call->args_.empty() ||
        !IsDenseVecTile(As<TileType>(assign->var_->GetType()))) {
      return Decline(reason, "loads require unbound, full, static 2D Vec tiles");
    }
    auto source = AsVarLike(call->args_[0]);
    if (!source || !tensor_params_.count(source.get())) {
      return Decline(reason, "load source is not a direct GM parameter");
    }
    for (size_t i = 1; i < call->args_.size(); ++i) {
      ExprReads arg_reads;
      arg_reads.VisitExpr(call->args_[i]);
      for (const auto* var : arg_reads.vars) {
        if (!As<ScalarType>(var->GetType()) || var->GetKind() == ObjectKind::IterArg) {
          return Decline(reason, "load address depends on non-scalar or carried data");
        }
      }
    }
    reads.insert(source.get());
    plan.loads.insert(assign->var_.get());
    DeclareRegion(plan, assign->var_);
    return true;
  }

  void DeclareRegion(LoopPlan& plan, const VarPtr& owner, int64_t slots = 0) {
    auto nested = plan.nesting.storage.find(owner.get());
    if (nested != plan.nesting.storage.end()) {
      plan.slot_axes[owner.get()] = nested->second;
      slots = plan.slots * nested->second.storage_stride;
      auto shared = plan.nested_owners.find(nested->second.family);
      if (shared != plan.nested_owners.end()) {
        const auto& first = plan.slot_axes.at(shared->second);
        if (first.storage_stride != nested->second.storage_stride || first.slots != nested->second.slots ||
            !CompatibleSlotTypes(owner->GetType(), shared->second->GetType())) {
          plan.nested_storage_valid = false;
        }
        plan.slot_roots[owner.get()] = shared->second;
        return;
      }
      plan.nested_owners[nested->second.family] = owner.get();
    }
    plan.slot_roots[owner.get()] = owner.get();
    plan.region_slots[owner.get()] = slots ? slots : plan.slots;
    plan.bases[owner.get()] = std::make_shared<Var>(
        "pipe_" + owner->name_hint_ + "_" + std::to_string(next_region_++), GetPtrType(), owner->span_);
  }

  static bool AnalyzeStore(const AssignStmtPtr& assign, const CallPtr& call,
                           const std::set<const Var*>& tensor_params, std::map<const Var*, const Var*>& roots,
                           std::set<const Var*>& writes, std::string& reason) {
    auto alias_arg = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
    if (!IsOp(call, "tile.store") || !alias_arg || !As<TensorType>(assign->var_->GetType())) {
      return Decline(reason, "only ordinary tile.store writes are supported");
    }
    auto target = AsVarLike(call->args_[*alias_arg]);
    if (!target) return Decline(reason, "store target has unresolved tensor identity");
    const auto known = roots.find(target.get());
    const Var* root = known != roots.end()                ? known->second
                      : tensor_params.count(target.get()) ? target.get()
                                                          : nullptr;
    if (!root) return Decline(reason, "store target has unresolved tensor identity");
    const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
    if (entry.GetArgEffect(*alias_arg, call->kwargs_) != ArgEffect::Write) {
      return Decline(reason, "atomic and read-modify-write stores are unsupported");
    }
    roots[assign->var_.get()] = root;
    writes.insert(root);
    return true;
  }

  bool AnalyzeCompute(const AssignStmtPtr& assign, const CallPtr& call,
                      const std::set<const Var*>& local_tiles, LoopPlan& plan,
                      std::set<const Var*>& scratch_tiles, std::set<const Var*>& read_tiles,
                      std::string& reason) const {
    auto definition = [&](const Var* root) -> CallPtr {
      auto scoped = plan.scope_definitions.find(root);
      if (scoped != plan.scope_definitions.end()) return scoped->second;
      auto original = definitions_.calls.find(root);
      return original == definitions_.calls.end() ? nullptr : original->second;
    };
    const bool view = plan.aliases.count(assign->var_.get()) != 0;
    if (!((plan.fifo_slots || plan.nesting.nested) && view) &&
        !IsDenseVecTile(As<TileType>(assign->var_->GetType()))) {
      return Decline(reason, "compute result requires an unbound full static Vec tile");
    }
    const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
    if (plan.nesting.nested && IsOp(call, "tile.create") &&
        entry.GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::NoAccess) {
      plan.local_workspace.insert(assign->var_.get());
      return true;
    }
    if ((!view && entry.GetExecutionMemoryAccessEvidence() != ExecutionMemoryAccessEvidence::Functional &&
         !entry.HasDeclaredArgEffects()) ||
        entry.IsNoDuplicate() || entry.GetCrossCoreRole() ||
        (!view && op_predicates::OutputInheritsSourceBuffer(call->op_->name_))) {
      return Decline(reason, "compute lacks a functional execution-memory contract");
    }
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto effect = entry.GetArgEffect(i, call->kwargs_);
      // Legacy reductions already declare hardware scratch through the lane
      // contract. Reuse that evidence without adding a second registration that
      // would change allocation constraints for every non-pipelined caller.
      const bool scratch =
          (entry.IsWorkspaceArg(i) && effect == ArgEffect::Write) ||
          (!entry.HasDeclaredArgEffect(i) && entry.GetLaneInvariantArgKind(i) == LaneInvariantArg::Scratch);
      if (effect != ArgEffect::Read && !scratch) {
        return Decline(reason, "compute mutates non-workspace input data");
      }
      ExprReads reads;
      reads.VisitExpr(call->args_[i]);
      for (const auto* var : reads.vars) {
        if (As<TileType>(var->GetType())) {
          const auto* root = AliasRoot(plan, var);
          if (scratch) {
            auto def = definition(root);
            if (!def || !IsOp(def, "tile.create") || !IsDenseVecTile(As<TileType>(root->GetType()))) {
              return Decline(reason, "workspace must own an unbound full tile.create allocation");
            }
            scratch_tiles.insert(root);
          } else if (!view) {
            read_tiles.insert(root);
          }
          if (!local_tiles.count(var)) {
            auto def = definition(root);
            if (!def || op_predicates::OutputInheritsSourceBuffer(def->op_->name_) ||
                !IsDenseVecTile(As<TileType>(var->GetType()))) {
              return Decline(reason, "external tile has unresolved storage ownership");
            }
          }
        } else if (!As<ScalarType>(var->GetType())) {
          return Decline(reason, "compute consumes a non-tile memory object");
        }
      }
    }
    return true;
  }

  static ExprPtr SourceIndex(const LoopPlan& plan, const ExprPtr& logical, const Span& span) {
    if (auto constant = As<ConstInt>(logical)) {
      return MakeConstIndex(plan.start + constant->value_ * plan.step, span);
    }
    ExprPtr index = logical;
    if (plan.step != 1) index = MakeMul(index, MakeConstIndex(plan.step, span), span);
    return plan.start == 0 ? index : MakeAdd(MakeConstIndex(plan.start, span), index, span);
  }

  static TypePtr SlotType(const TypePtr& type, const VarPtr& base, const ExprPtr& logical, int64_t slots,
                          const Span& span, const ExprPtr& explicit_slot = nullptr) {
    ExprPtr index;
    if (explicit_slot) {
      index = explicit_slot;
    } else if (auto constant = As<ConstInt>(logical)) {
      index = MakeConstIndex(constant->value_ % slots, span);
    } else {
      index = MakeFloorMod(logical, MakeConstIndex(slots, span), span);
    }
    auto memref = std::make_shared<MemRef>(base, int64_t{0}, uint64_t{0}, span, true,
                                           static_cast<uint64_t>(slots), std::make_optional(index));
    return CloneTypeWithMemRef(type, std::optional<MemRefPtr>(memref));
  }

  struct Iteration {
    std::vector<StmtPtr> stmts;
    std::vector<ExprPtr> yields;
    std::unordered_set<const Var*> live_outputs;
  };

  using AxisKey = std::tuple<int64_t, int64_t, int64_t>;

  static Iteration CloneIteration(const ForStmtPtr& op, const LoopPlan& plan, const ExprPtr& logical,
                                  const std::vector<ExprPtr>& carries, bool producer,
                                  const ExprPtr& explicit_slot = nullptr,
                                  const std::map<AxisKey, ExprPtr>& scoped_slots = {},
                                  const std::set<const Var*>* selected_loads = nullptr,
                                  const std::map<const Var*, std::vector<StmtPtr>>& issues = {}) {
    std::unordered_map<const Var*, ExprPtr> seeds{
        {op->loop_var_.get(), SourceIndex(plan, logical, op->span_)}};
    for (size_t i = 0; i < op->iter_args_.size(); ++i) seeds[op->iter_args_[i].get()] = carries[i];
    auto clone = DeepClone(op->body_, seeds);
    std::unordered_map<const Var*, ExprPtr> replacements;
    std::map<const Var*, const Var*> originals;
    for (const auto& [original, fresh] : clone.var_map) {
      originals[fresh.get()] = original;
      auto root = plan.slot_roots.find(original);
      // InitMemRef derives mandatory views from their source. Binding the view
      // itself would incorrectly turn it into an independent allocation.
      if (root != plan.slot_roots.end() && !plan.aliases.count(original)) {
        const auto count = plan.region_slots.at(root->second);
        auto slot = explicit_slot;
        if (!scoped_slots.empty()) {
          auto axis = plan.slot_axes.find(original);
          AxisKey key = axis == plan.slot_axes.end()
                            ? AxisKey{1, 0, count}
                            : AxisKey{axis->second.storage_stride, axis->second.storage_phase, count};
          slot = scoped_slots.at(key);
        }
        auto type =
            SlotType(fresh->GetType(), plan.bases.at(root->second), logical, count, fresh->span_, slot);
        auto replacement = std::make_shared<Var>(fresh->name_hint_, type, fresh->span_);
        replacements[fresh.get()] = replacement;
        originals[replacement.get()] = original;
      }
    }
    // Rewrite the whole clone once. Constructing the substitution mutator for
    // every assignment would copy an O(N) map O(N) times.
    auto rebound = transform_utils::Substitute(clone.cloned_body, replacements);
    auto [body, yields] = SplitBodyYield(rebound);
    auto seq = As<SeqStmts>(body);
    INTERNAL_CHECK_SPAN(seq, op->span_) << "Internal error: cloned software pipeline body must be a sequence";
    Iteration result;
    std::function<StmtPtr(const StmtPtr&)> filter = [&](const StmtPtr& stmt) -> StmtPtr {
      if (auto sequence = As<SeqStmts>(stmt)) {
        std::vector<StmtPtr> kept;
        for (const auto& child : sequence->stmts_) {
          if (auto selected = filter(child)) kept.push_back(selected);
        }
        return SeqStmts::Flatten(std::move(kept), sequence->span_);
      }
      if (auto branch = As<IfStmt>(stmt)) {
        return std::make_shared<IfStmt>(branch->condition_, filter(branch->then_body_), std::nullopt,
                                        std::vector<VarPtr>{}, branch->span_);
      }
      // The owned receive lowers to pop + load + free of the GM entry.
      // Its UB lifetime is independent of that entry.
      if (As<EvalStmt>(stmt)) return nullptr;
      if (auto yield = As<YieldStmt>(stmt); yield && yield->value_.empty()) {
        return nullptr;
      }
      auto assign = As<AssignStmt>(stmt);
      INTERNAL_CHECK_SPAN(assign, op->span_)
          << "Internal error: eligible pipeline statement must be an assignment";
      const auto* original = originals.at(assign->var_.get());
      if (plan.nesting.anchors.count(original)) {
        auto issue = issues.find(original);
        return !producer && issue != issues.end() ? SeqStmts::Flatten(issue->second, assign->span_) : nullptr;
      }
      const bool load = plan.loads.count(original) != 0;
      auto axis = plan.slot_axes.find(original);
      const bool staged = axis == plan.slot_axes.end() || axis->second.phase < axis->second.slots;
      if (producer && !load && !plan.scalars.count(original)) return nullptr;
      if (producer && load && (selected_loads ? !selected_loads->count(original) : !staged)) return nullptr;
      if (producer && load) result.live_outputs.insert(assign->var_.get());
      if (!producer && load && (!plan.nesting.nested || plan.prefetched_loads.count(original))) {
        auto type = As<TileType>(assign->var_->GetType());
        const auto& span = assign->span_;
        auto shape = std::make_shared<MakeTuple>(type->shape_, span);
        auto call = std::make_shared<Call>(OpRegistry::GetInstance().GetOp("tile.create"),
                                           std::vector<ExprPtr>{shape},
                                           std::vector<std::pair<std::string, std::any>>{
                                               {"dtype", type->dtype_}, {"target_memory", MemorySpace::Vec}},
                                           type, span);
        auto marked = call;
        marked->attrs_.emplace_back(kSoftwarePipelineSlotsAttr, true);
        return std::make_shared<AssignStmt>(assign->var_, marked, span);
      }
      if (load) {
        auto marked = MutableCopy(As<Call>(assign->value_));
        marked->attrs_.emplace_back(kSoftwarePipelineSlotsAttr, true);
        return std::make_shared<AssignStmt>(assign->var_, marked, assign->span_);
      }
      return assign;
    };
    for (const auto& stmt : seq->stmts_) {
      if (auto selected = filter(stmt)) result.stmts.push_back(selected);
    }
    if (!producer) result.yields = std::move(yields);
    return result;
  }

  /// Child residues select contiguous banks; each bank rotates root stages.
  /// This is the same mixed-radix hierarchy with the root coordinate stored
  /// fastest: bank * root_stage + (root_iteration % root_stage). No trip count
  /// enters storage indexing, and every bank index dominates backend waits.
  static std::map<AxisKey, ExprPtr> ScopedSlots(const LoopPlan& plan, const ExprPtr& iv,
                                                int64_t parent_distance, std::vector<StmtPtr>& body,
                                                std::map<int64_t, ExprPtr>& ordinals) {
    const auto& span = plan.nesting.loop->span_;
    std::map<AxisKey, ExprPtr> slots;
    std::set<AxisKey> axes;
    for (const auto& [var, owner] : plan.slot_roots) {
      auto count = plan.region_slots.at(owner);
      auto axis = plan.slot_axes.find(var);
      AxisKey key = axis == plan.slot_axes.end()
                        ? AxisKey{1, 0, count}
                        : AxisKey{axis->second.storage_stride, axis->second.storage_phase, count};
      axes.insert(key);
    }
    for (const auto& key : axes) {
      auto [stride, phase, depth] = key;
      const auto root_depth = depth / stride;
      if (auto constant = As<ConstInt>(iv)) {
        slots.emplace(key, MakeConstIndex(
                               phase * root_depth + (constant->value_ + parent_distance) % root_depth, span));
        continue;
      }
      // Within a plan every bank has the same root depth. Cache by issue distance.
      if (!ordinals.count(parent_distance)) {
        ExprPtr logical = iv;
        if (parent_distance) logical = MakeAdd(iv, MakeConstIndex(parent_distance, span), span);
        auto ordinal = std::make_shared<Var>("parent_slot", iv->GetType(), span);
        body.push_back(std::make_shared<AssignStmt>(
            ordinal, MakeFloorMod(logical, MakeConstIndex(root_depth, span), span), span));
        ordinals.emplace(parent_distance, ordinal);
      }
      ExprPtr index = ordinals.at(parent_distance);
      if (phase) {
        auto slot = std::make_shared<Var>("nested_slot", iv->GetType(), span);
        body.push_back(std::make_shared<AssignStmt>(
            slot, MakeAdd(index, MakeConstIndex(phase * root_depth, span), span), span));
        index = slot;
      }
      slots.emplace(key, index);
    }
    return slots;
  }

  /// Preserve one outer loop and issue each stream's planned producer slices,
  /// across parent boundaries. Only producer slices cross those boundaries;
  /// parent compute and side effects keep their original execution order.
  static StmtPtr BuildScoped(const ForStmtPtr& op, const LoopPlan& plan) {
    const auto& span = op->span_;
    const auto zero = MakeConstIndex(0, span);
    auto iv = std::make_shared<Var>(op->loop_var_->name_hint_ + "_pipe", op->loop_var_->GetType(), span);
    std::vector<StmtPtr> result;
    auto initial_carries = pipeline_loop::InitValueExprs(op->iter_args_);
    for (const auto& [distance, selected] : plan.prologue_loads) {
      std::vector<StmtPtr> prelude;
      std::map<int64_t, ExprPtr> ordinals;
      auto slots = ScopedSlots(plan, zero, distance, prelude, ordinals);
      auto logical = MakeConstIndex(distance, span);
      auto preload = CloneIteration(op, plan, logical, initial_carries, true, nullptr, slots, &selected);
      prelude.insert(prelude.end(), preload.stmts.begin(), preload.stmts.end());
      prelude = dce::EliminateDeadCode(prelude, preload.live_outputs);
      result.push_back(std::make_shared<IfStmt>(MakeLt(logical, op->stop_, span),
                                                SeqStmts::Flatten(std::move(prelude), span), std::nullopt,
                                                std::vector<VarPtr>{}, span));
    }
    std::vector<StmtPtr> body;
    std::map<int64_t, ExprPtr> ordinals;
    std::map<int64_t, std::map<AxisKey, ExprPtr>> slots_by_distance;
    slots_by_distance.emplace(0, ScopedSlots(plan, iv, 0, body, ordinals));
    std::vector<IterArgPtr> args;
    std::vector<ExprPtr> carries;
    for (const auto& original : op->iter_args_) {
      auto arg = pipeline_loop::MakeFreshIterArg(original, original->initValue_);
      args.push_back(arg);
      carries.push_back(arg);
    }
    auto inputs =
        CloneIteration(op, plan, iv, carries, true, nullptr, slots_by_distance.at(0), &plan.head_loads);
    inputs.stmts = dce::EliminateDeadCode(inputs.stmts, inputs.live_outputs);
    std::map<const Var*, std::vector<StmtPtr>> issues;
    std::unordered_set<const Var*> future_live;
    // There are at most 64 phase anchors. Slice each issue group with the
    // existing clone/effect/DCE machinery: bounded O(N log N), not one unbounded
    // whole-body scan per operation or per runtime iteration.
    for (const auto& [anchor, groups] : plan.prefetch_issues) {
      auto& at_anchor = issues[anchor];
      for (const auto& [distance, selected] : groups) {
        if (!slots_by_distance.count(distance)) {
          slots_by_distance.emplace(distance, ScopedSlots(plan, iv, distance, body, ordinals));
        }
        auto logical = distance ? MakeAdd(iv, MakeConstIndex(distance, span), span) : ExprPtr(iv);
        auto future = CloneIteration(op, plan, logical, carries, true, nullptr,
                                     slots_by_distance.at(distance), &selected);
        future.stmts = dce::EliminateDeadCode(future.stmts, future.live_outputs);
        future_live.insert(future.live_outputs.begin(), future.live_outputs.end());
        if (distance) {
          at_anchor.push_back(std::make_shared<IfStmt>(MakeLt(logical, op->stop_, span),
                                                       SeqStmts::Flatten(std::move(future.stmts), span),
                                                       std::nullopt, std::vector<VarPtr>{}, span));
        } else {
          at_anchor.insert(at_anchor.end(), future.stmts.begin(), future.stmts.end());
        }
      }
    }
    auto current =
        CloneIteration(op, plan, iv, carries, false, nullptr, slots_by_distance.at(0), nullptr, issues);
    current.stmts = dce::EliminateDeadCode(current.stmts, future_live);
    body.insert(body.end(), inputs.stmts.begin(), inputs.stmts.end());
    if (auto head = issues.find(nullptr); head != issues.end()) {
      body.insert(body.end(), head->second.begin(), head->second.end());
    }
    body.insert(body.end(), current.stmts.begin(), current.stmts.end());
    if (!args.empty()) body.push_back(std::make_shared<YieldStmt>(current.yields, span));
    future_live.insert(inputs.live_outputs.begin(), inputs.live_outputs.end());
    body = dce::EliminateDeadCode(body, future_live);
    std::vector<std::pair<std::string, std::any>> attrs{
        {kSoftwarePipelineSlotsAttr, static_cast<int>(plan.slots)}};
    if (plan.nesting.nested) {
      attrs.emplace_back(kSoftwarePipelineNestedAttr, static_cast<int>(plan.nesting.max_stride));
    }
    result.push_back(std::make_shared<ForStmt>(iv, zero, op->stop_, MakeConstIndex(1, span), args,
                                               SeqStmts::Flatten(std::move(body), span), op->return_vars_,
                                               span, ForKind::Sequential, attrs));
    return SeqStmts::Flatten(std::move(result), span);
  }

  /// One guarded body for runtime trip counts: no independently allocated tail.
  /// Slot ordinals are materialized outside the guard so backend-inserted waits
  /// dominate both branches. Address computations and loads stay inside it.
  static StmtPtr BuildPositiveDynamic(const ForStmtPtr& op, const LoopPlan& plan) {
    const auto& span = op->span_;
    const auto zero = MakeConstIndex(0, span);
    const auto one = MakeConstIndex(1, span);
    const auto start = MakeConstIndex(plan.start, span);
    const auto step = MakeConstIndex(plan.step, span);
    const auto distance = MakeConstIndex(plan.slots - 1, span);
    auto extent = MakeSub(op->stop_, start, span);
    // ceil(extent / step) without extent + step - 1 overflow.
    auto count =
        MakeAdd(MakeFloorDiv(extent, step, span), MakeMin(MakeFloorMod(extent, step, span), one, span), span);
    auto trips = std::make_shared<Var>(op->loop_var_->name_hint_ + "_trips", op->loop_var_->GetType(), span);
    std::vector<StmtPtr> result;
    auto carries = pipeline_loop::InitValueExprs(op->iter_args_);
    for (int64_t i = 0; i < plan.slots - 1; ++i) {
      auto ordinal = MakeConstIndex(i, span);
      auto preload = CloneIteration(op, plan, ordinal, carries, true);
      result.push_back(std::make_shared<IfStmt>(MakeLt(ordinal, trips, span),
                                                SeqStmts::Flatten(std::move(preload.stmts), span),
                                                std::nullopt, std::vector<VarPtr>{}, span));
    }
    auto iv = std::make_shared<Var>(op->loop_var_->name_hint_ + "_pipe", op->loop_var_->GetType(), span);
    std::vector<IterArgPtr> iter_args;
    carries.clear();
    for (const auto& original : op->iter_args_) {
      auto arg = pipeline_loop::MakeFreshIterArg(original, original->initValue_);
      iter_args.push_back(arg);
      carries.push_back(arg);
    }
    auto current = CloneIteration(op, plan, iv, carries, false);
    // Keep the canonical iv + lead spelling recognized by backend rotation.
    // An explicit bounded fast path proves this addition cannot overflow;
    // larger counts retain the original sequential loop below.
    auto more = MakeLt(iv, MakeSub(trips, distance, span), span);
    auto future = MakeAdd(iv, distance, span);
    auto slot =
        std::make_shared<Var>(op->loop_var_->name_hint_ + "_next_slot", op->loop_var_->GetType(), span);
    auto next = CloneIteration(op, plan, future, carries, true, slot);
    std::vector<StmtPtr> body{std::make_shared<AssignStmt>(
                                  slot, MakeFloorMod(future, MakeConstIndex(plan.slots, span), span), span),
                              std::make_shared<IfStmt>(more, SeqStmts::Flatten(std::move(next.stmts), span),
                                                       std::nullopt, std::vector<VarPtr>{}, span)};
    body.insert(body.end(), current.stmts.begin(), current.stmts.end());
    if (!iter_args.empty()) body.push_back(std::make_shared<YieldStmt>(current.yields, span));
    auto limit = MakeConstIndex(std::numeric_limits<int64_t>::max() - plan.slots, span);
    auto fast_returns = plan.needs_overflow_fallback
                            ? pipeline_loop::MakeFreshReturnVars(op->return_vars_, "_pipeline")
                            : op->return_vars_;
    result.push_back(std::make_shared<ForStmt>(
        iv, zero, MakeMin(trips, limit, span), one, iter_args, SeqStmts::Flatten(std::move(body), span),
        fast_returns, span, ForKind::Sequential,
        std::vector<std::pair<std::string, std::any>>{
            {kSoftwarePipelineSlotsAttr, static_cast<int>(plan.slots)}}));
    if (!plan.needs_overflow_fallback) {
      result.insert(result.begin(), std::make_shared<AssignStmt>(trips, count, span));
      return SeqStmts::Flatten(std::move(result), span);
    }
    if (!fast_returns.empty()) {
      result.push_back(std::make_shared<YieldStmt>(pipeline_loop::ReturnVarsAsExprs(fast_returns), span));
    }
    auto cloned = DeepClone(op);
    auto slow = MutableCopy(As<ForStmt>(cloned.cloned_body));
    slow->kind_ = ForKind::Sequential;
    slow->attrs_ = StripAttr(slow->attrs_, kPipelineStagesAttr);
    std::vector<StmtPtr> fallback{slow};
    if (!slow->return_vars_.empty()) {
      fallback.push_back(
          std::make_shared<YieldStmt>(pipeline_loop::ReturnVarsAsExprs(slow->return_vars_), span));
    }
    return SeqStmts::Flatten(
        {std::make_shared<AssignStmt>(trips, count, span),
         std::make_shared<IfStmt>(MakeLe(trips, limit, span), SeqStmts::Flatten(std::move(result), span),
                                  SeqStmts::Flatten(std::move(fallback), span), op->return_vars_, span)},
        span);
  }

  static StmtPtr BuildDynamic(const ForStmtPtr& op, const LoopPlan& plan) {
    const auto& span = op->span_;
    auto positive = MutableCopy(op);
    positive->return_vars_ = pipeline_loop::MakeFreshReturnVars(op->return_vars_, "_nonempty");
    std::vector<StmtPtr> then_stmts{BuildPositiveDynamic(positive, plan)};
    std::optional<StmtPtr> else_body;
    if (!op->return_vars_.empty()) {
      then_stmts.push_back(
          std::make_shared<YieldStmt>(pipeline_loop::ReturnVarsAsExprs(positive->return_vars_), span));
      else_body = std::make_shared<YieldStmt>(pipeline_loop::InitValueExprs(op->iter_args_), span);
    }
    // Subtract only in the nonempty domain. Computing max(stop-start, 0)
    // outside this guard overflows for stop=INT64_MIN and a positive start.
    return std::make_shared<IfStmt>(MakeGt(op->stop_, MakeConstIndex(plan.start, span), span),
                                    SeqStmts::Flatten(std::move(then_stmts), span), else_body,
                                    op->return_vars_, span);
  }

  static StmtPtr Build(const ForStmtPtr& op, const LoopPlan& plan) {
    if (plan.dynamic_trips) return BuildDynamic(op, plan);
    const auto& span = op->span_;
    const int64_t prefetch = plan.slots - 1;
    std::vector<StmtPtr> result;
    auto carries = pipeline_loop::InitValueExprs(op->iter_args_);
    for (int64_t i = 0; i < prefetch; ++i) {
      auto preload = CloneIteration(op, plan, MakeConstIndex(i, span), carries, true);
      result.insert(result.end(), preload.stmts.begin(), preload.stmts.end());
    }
    if (plan.trips > prefetch) {
      auto iv = std::make_shared<Var>(op->loop_var_->name_hint_ + "_pipe", op->loop_var_->GetType(), span);
      std::vector<IterArgPtr> iter_args;
      std::vector<ExprPtr> steady_carries;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        auto arg = pipeline_loop::MakeFreshIterArg(op->iter_args_[i], carries[i]);
        iter_args.push_back(arg);
        steady_carries.push_back(arg);
      }
      auto next = CloneIteration(op, plan, OffsetIndex(iv, prefetch, span), steady_carries, true);
      auto current = CloneIteration(op, plan, iv, steady_carries, false);
      next.stmts.insert(next.stmts.end(), current.stmts.begin(), current.stmts.end());
      if (!iter_args.empty()) next.stmts.push_back(std::make_shared<YieldStmt>(current.yields, span));
      auto returns = pipeline_loop::MakeFreshReturnVars(op->return_vars_, "_steady");
      result.push_back(std::make_shared<ForStmt>(
          iv, MakeConstIndex(0, span), MakeConstIndex(plan.trips - prefetch, span), MakeConstIndex(1, span),
          iter_args, SeqStmts::Flatten(std::move(next.stmts), span), returns, span, ForKind::Sequential));
      carries = pipeline_loop::ReturnVarsAsExprs(returns);
    }
    for (int64_t i = plan.trips - prefetch; i < plan.trips; ++i) {
      auto drain = CloneIteration(op, plan, MakeConstIndex(i, span), carries, false);
      result.insert(result.end(), drain.stmts.begin(), drain.stmts.end());
      carries = std::move(drain.yields);
    }
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      result.push_back(std::make_shared<AssignStmt>(op->return_vars_[i], carries[i], span));
    }
    return SeqStmts::Flatten(std::move(result), span);
  }

  std::set<const Var*> tensor_params_;
  IOCategoryOps io_;
  TileDefinitions definitions_;
  arith::Analyzer scalar_bounds_;
  int next_region_ = 0;
  bool inside_pipeline_ = false;
  const ForStmt* fifo_loop_ = nullptr;
  int fifo_slots_ = 0;
};

}  // namespace

FunctionPtr LowerSoftwarePipeline(const FunctionPtr& func) {
  SoftwarePipelineMutator mutator(func);
  auto body = mutator.VisitStmt(func->body_);
  if (!mutator.changed) return func;
  auto result = MutableCopy(func);
  result->body_ = body;
  return result;
}

FunctionPtr LowerFifoSoftwarePipeline(const FunctionPtr& func, const ForStmtPtr& loop, int fifo_slots) {
  SoftwarePipelineMutator mutator(func, loop.get(), fifo_slots);
  auto body = mutator.VisitStmt(func->body_);
  if (!mutator.changed) return func;
  auto result = MutableCopy(func);
  result->body_ = body;
  return result;
}

}  // namespace pypto::ir
