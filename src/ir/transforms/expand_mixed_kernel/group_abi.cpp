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

#include "src/ir/transforms/expand_mixed_kernel/group_abi.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/call_shape_binding.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace expand_mixed_kernel {

namespace {

using loop_repair::BuildDefMap;
const auto& FlattenBody = transform_utils::FlattenToStmts;

// ============================================================================
// Hand-written Group ABI normalization
// ============================================================================

/// Runtime MixedKernels subslots share one CoreTaskArgs payload. Auto-expanded
/// Groups already satisfy that contract because both member calls forward the
/// complete Group signature. A hand-written Group may call AIC/AIV functions
/// with different subsets, however, so normalize both members to the Group ABI
/// before orchestration codegen (#2097).

class KernelCallCounter : public IRVisitor {
 public:
  [[nodiscard]] size_t Count(const std::string& name) const {
    auto it = counts_.find(name);
    return it == counts_.end() ? 0 : it->second;
  }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    CountCallee(op->op_);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    CountCallee(op->op_);
    IRVisitor::VisitExpr_(op);
  }

 private:
  void CountCallee(const OpPtr& op) {
    if (auto gv = std::dynamic_pointer_cast<const GlobalVar>(op)) ++counts_[gv->name_];
  }

  std::unordered_map<std::string, size_t> counts_;
};

/// Ascend910 no-split cross-core pipes need both AIV sub-lanes to execute:
/// lane 0 performs the real work while SplitVectorKernel synthesizes a
/// zero-valid-shape replay on lane 1 so every hardware pipe handshake is
/// balanced. Auto-expanded mixed kernels are stamped in ExpandMixedFunction;
/// hand-written AIC/AIV Groups need the same inference here.
class NoSplitCrossCoreTransportCollector : public IRVisitor {
 public:
  [[nodiscard]] bool UsesNoSplitTransportOnly() const { return uses_transport_ && !uses_split_transport_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (op_predicates::IsTPush(op) || op_predicates::IsTPop(op) || op_predicates::IsTFree(op)) {
      uses_transport_ = true;
      uses_split_transport_ = uses_split_transport_ || op->GetKwarg<int>("split", 0) != 0;
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  bool uses_transport_ = false;
  bool uses_split_transport_ = false;
};

bool NeedsInferredNoSplitDualAivDispatch(const FunctionPtr& func) {
  if (!func || func->func_type_ != FunctionType::AIV || !pypto::backend::BackendConfig::IsConfigured()) {
    return false;
  }
  const auto* pass_context = PassContext::Current();
  const auto* backend_handler = pass_context ? pass_context->GetBackendHandler()
                                             : pypto::backend::BackendConfig::GetBackend()->GetHandler();
  if (!backend_handler->RequiresNoSplitDualAivDispatch() ||
      func->GetAttr<bool>(kAttrDualAivDispatch, false) || func->HasAttr(kAttrExternalSource) ||
      func->requires_runtime_binding_) {
    return false;
  }
  if (auto mode = func->GetSplitMode(); mode.has_value() && *mode != SplitMode::None) {
    return false;
  }

  NoSplitCrossCoreTransportCollector collector;
  collector.VisitStmt(func->body_);
  return collector.UsesNoSplitTransportOnly();
}

FunctionPtr WithDualAivDispatch(const FunctionPtr& func) {
  auto result = MutableCopy(func);
  auto attrs = result->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& kv) { return kv.first == kAttrDualAivDispatch; }),
              attrs.end());
  attrs.emplace_back(kAttrDualAivDispatch, true);
  result->attrs_ = std::move(attrs);
  return result;
}

bool IsCanonicalGroupMemberCall(const CallPtr& call, const FunctionPtr& callee, const FunctionPtr& group) {
  if (!call || !callee || !group) return false;
  if (call->args_.size() != group->params_.size() || callee->params_.size() != group->params_.size()) {
    return false;
  }
  for (size_t i = 0; i < group->params_.size(); ++i) {
    auto arg = AsVarLike(call->args_[i]);
    if (!arg || arg.get() != group->params_[i].get()) return false;
  }
  return true;
}

/// Rewrite import_peer_buffer(peer_func=...) when a reused pair receives
/// Group-private adapter names.
class PeerFuncRewriter : public IRMutator {
 public:
  explicit PeerFuncRewriter(std::unordered_map<std::string, std::string> rename_map)
      : rename_map_(std::move(rename_map)) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = As<Call>(IRMutator::VisitExpr_(op));
    INTERNAL_CHECK_SPAN(visited != nullptr, op->span_) << "Call mutation produced a non-Call expression";

    auto builtin = As<Op>(visited->op_);
    if (!builtin || !IsOp(builtin, "system.import_peer_buffer")) return visited;

    auto kwargs = visited->kwargs_;
    bool changed = false;
    for (auto& [key, value] : kwargs) {
      if (key != "peer_func") continue;
      auto peer = std::any_cast<std::string>(&value);
      if (!peer) continue;
      auto it = rename_map_.find(*peer);
      if (it == rename_map_.end()) continue;
      value = it->second;
      changed = true;
    }
    if (!changed) return visited;
    return std::make_shared<Call>(visited->op_, visited->args_, std::move(kwargs), visited->attrs_,
                                  visited->GetType(), visited->span_);
  }

 private:
  std::unordered_map<std::string, std::string> rename_map_;
};

using GroupAbiArgIndices = std::unordered_map<const Expr*, int32_t>;

using CalleeOutputSummaries =
    std::unordered_map<const Function*, std::vector<return_lineage::ReturnedValueBinding>>;

// Preserve output identity before asking whether its storage aliases a Group
// parameter. An unknown param_index is still a real producer output: it must
// be materialized by tensor-to-tile lowering, not silently captured by a peer.
// Summaries are callee-relative and shared across Groups; producer instances
// and SSA definitions are indexed separately for each Group call site.
class GroupOutputBindings {
 public:
  struct Output {
    FunctionPtr callee;
    size_t position = 0;
    const return_lineage::ReturnedValueBinding* binding;
  };

  GroupOutputBindings(const FunctionPtr& group, ProgramPtr program, CalleeOutputSummaries& summaries)
      : program_(std::move(program)), summaries_(summaries) {
    BuildDefMap(FlattenBody(group->body_), definitions_);
    class CallCollector : public IRVisitor {
     public:
      explicit CallCollector(GroupOutputBindings& owner) : owner_(owner) {}

     protected:
      void VisitExpr_(const CallPtr& call) override {
        owner_.RecordProducer(call, call->op_);
        IRVisitor::VisitExpr_(call);
      }
      void VisitExpr_(const SubmitPtr& submit) override {
        owner_.RecordProducer(submit, submit->op_);
        IRVisitor::VisitExpr_(submit);
      }

     private:
      GroupOutputBindings& owner_;
    } collector(*this);
    collector.VisitStmt(group->body_);
  }

  const Output* Find(const ExprPtr& expr) {
    if (!expr) return nullptr;
    if (auto it = outputs_.find(expr.get()); it != outputs_.end()) {
      return it->second;
    }
    // A provisional miss guards malformed cyclic definitions. Alias prefixes
    // are compressed so repeated arguments do not rescan their SSA chains.
    std::vector<const Expr*> path;
    ExprPtr current = expr;
    const Output* output = nullptr;
    while (current) {
      if (auto it = outputs_.find(current.get()); it != outputs_.end()) {
        output = it->second;
        break;
      }
      outputs_.emplace(current.get(), nullptr);
      path.push_back(current.get());
      if (auto var = AsVarLike(current)) {
        auto it = definitions_.find(var.get());
        auto definition = it != definitions_.end() ? As<AssignStmt>(it->second) : nullptr;
        current = definition ? definition->value_ : nullptr;
      } else if (auto projection = As<TupleGetItemExpr>(current)) {
        auto source = TupleSource(projection->tuple_);
        if (auto tuple = As<MakeTuple>(source)) {
          const auto pos = projection->index_;
          current = pos >= 0 && static_cast<size_t>(pos) < tuple->elements_.size()
                        ? tuple->elements_[static_cast<size_t>(pos)]
                        : nullptr;
        } else {
          output = At(source, projection->index_);
          break;
        }
      } else {
        // A direct single-result call has output position zero. A whole
        // multi-result tuple intentionally has no single output identity.
        auto it = producers_.find(current.get());
        if (it != producers_.end() && it->second.outputs.size() == 1) output = At(current, 0);
        break;
      }
    }
    for (const auto* item : path) outputs_[item] = output;
    return output;
  }

 private:
  struct Producer {
    std::vector<Output> outputs;
  };

  void RecordProducer(const ExprPtr& call, const OpPtr& op) {
    if (producers_.count(call.get())) return;
    auto global = As<GlobalVar>(op);
    auto callee = global ? program_->GetFunction(global->name_) : nullptr;
    if (!callee) return;
    auto it = summaries_.find(callee.get());
    if (it == summaries_.end()) {
      it = summaries_.emplace(callee.get(), return_lineage::AnalyzeReturnedValues(callee, program_)).first;
    }
    std::vector<Output> outputs;
    outputs.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) outputs.push_back({callee, i, &it->second[i]});
    producers_.emplace(call.get(), Producer{std::move(outputs)});
  }

  const Output* At(const ExprPtr& expr, int64_t position) const {
    if (!expr || position < 0) return nullptr;
    auto it = producers_.find(expr.get());
    if (it == producers_.end() || static_cast<size_t>(position) >= it->second.outputs.size()) {
      return nullptr;
    }
    // A plain Call returning one TupleType has one logical output, but its
    // tuple[i] selects an element inside that output. Submit instead wraps
    // each declared output in an outer tuple alongside its TaskId.
    if (As<Call>(expr) && it->second.outputs.size() == 1 &&
        As<TupleType>(it->second.outputs[0].binding->type)) {
      return nullptr;
    }
    size_t pos = static_cast<size_t>(position);
    return &it->second.outputs[pos];
  }

  ExprPtr TupleSource(ExprPtr expr) {
    std::vector<const Expr*> path;
    auto finish = [&](const ExprPtr& source) {
      for (const auto* item : path) tuple_sources_[item] = source;
      return source;
    };
    while (expr) {
      if (auto it = tuple_sources_.find(expr.get()); it != tuple_sources_.end()) {
        return finish(it->second);
      }
      tuple_sources_.emplace(expr.get(), nullptr);
      path.push_back(expr.get());
      if (auto var = AsVarLike(expr)) {
        auto it = definitions_.find(var.get());
        auto definition = it != definitions_.end() ? As<AssignStmt>(it->second) : nullptr;
        expr = definition ? definition->value_ : nullptr;
      } else if (auto projection = As<TupleGetItemExpr>(expr)) {
        auto source = TupleSource(projection->tuple_);
        auto tuple = As<MakeTuple>(source);
        if (!tuple || projection->index_ < 0 ||
            static_cast<size_t>(projection->index_) >= tuple->elements_.size()) {
          return finish(nullptr);
        }
        expr = tuple->elements_[static_cast<size_t>(projection->index_)];
      } else {
        return finish(expr);
      }
    }
    return finish(nullptr);
  }

  ProgramPtr program_;
  CalleeOutputSummaries& summaries_;
  std::unordered_map<const Var*, StmtPtr> definitions_;
  std::unordered_map<const Expr*, Producer> producers_;
  // Producer vectors are immutable after construction, so cached pointers
  // retain the exact call-site/output identity for this Group's lifetime.
  std::unordered_map<const Expr*, const Output*> outputs_;
  std::unordered_map<const Expr*, ExprPtr> tuple_sources_;
};

GroupAbiArgIndices BuildGroupAbiArgIndices(const FunctionPtr& group, const WrapperCallInfo& aic,
                                           const WrapperCallInfo& aiv, const ProgramPtr& program,
                                           GroupOutputBindings& outputs) {
  class ArgExprCollector : public IRVisitor {
   public:
    std::vector<ExprPtr> exprs;

    void VisitExpr(const ExprPtr& expr) override {
      if (!expr || !seen_.insert(expr.get()).second) return;
      if (AsVarLike(expr) || As<TupleGetItemExpr>(expr)) exprs.push_back(expr);
      IRVisitor::VisitExpr(expr);
    }

   protected:
    // The lineage index follows carry initializers; collect only operands here.
    void VisitExpr_(const IterArgPtr& /*var*/) override {}

   private:
    std::unordered_set<const Expr*> seen_;
  } collector;
  for (const auto& call : {aic.inner_call, aiv.inner_call}) {
    for (const auto& arg : call->args_) collector.VisitExpr(arg);
  }

  // Trace whole projections as well as SSA variables. A tuple has no single
  // parameter root, but each selected complete-parameter writeback does.
  // The lineage utility excludes views and fresh buffers from these aliases.
  // Preserve output identities before resolving their storage, including
  // outputs with no parameter lineage. Reuse the expression tracer for the
  // binding itself so full type/view checks and tuple boundaries stay in one
  // place rather than tracing a substituted source with weaker checks.
  for (const auto& expr : collector.exprs) outputs.Find(expr);
  auto indices =
      return_lineage::TraceExprsToParamIndices(collector.exprs, group->body_, group->params_, program);
  GroupAbiArgIndices result;
  result.reserve(group->params_.size() + collector.exprs.size());
  for (size_t i = 0; i < group->params_.size(); ++i) {
    result.emplace(group->params_[i].get(), static_cast<int32_t>(i));
  }
  for (size_t i = 0; i < collector.exprs.size(); ++i) {
    if (auto index = indices[i]) {
      result.emplace(collector.exprs[i].get(), static_cast<int32_t>(index.value()));
    }
  }
  return result;
}

// Replace a complete projection before visiting its tuple: the tuple belongs
// to the Group and must not become a captured free variable in the adapter.
class GroupAbiArgSubstituter : public IRMutator {
 public:
  GroupAbiArgSubstituter(const GroupAbiArgIndices& indices, std::vector<VarPtr>& params) {
    for (const auto& [expr, index] : indices) {
      var_remap_.emplace(expr, params[static_cast<size_t>(index)]);
    }
    // A fresh Tensor[n] still carries the original Group's n until its type
    // is visited. Publish the final type-remapped Var into the signature before
    // substituting arguments, so every use names that exact parameter identity.
    // All initial mappings are seeded first to support forward shape references.
    for (auto& param : params) {
      auto final_param = As<Var>(VisitExpr(param));
      INTERNAL_CHECK_SPAN(final_param != nullptr, param->span_)
          << "Internal error: Group adapter parameter remapped to a non-Var expression";
      param = std::move(final_param);
    }
    for (const auto& [expr, index] : indices) {
      var_remap_[expr] = params[static_cast<size_t>(index)];
    }
  }

 protected:
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    if (auto it = var_remap_.find(op.get()); it != var_remap_.end()) {
      return ResolveVarRemapHit(op.get(), it->second);
    }
    return IRMutator::VisitExpr_(op);
  }
};

FunctionPtr BuildGroupAbiAdapter(const FunctionPtr& group, const WrapperCallInfo& member,
                                 const std::string& adapter_name,
                                 const std::unordered_map<std::string, std::string>& peer_renames,
                                 const GroupAbiArgIndices& canonical_arg_indices,
                                 GroupOutputBindings& outputs) {
  const auto& callee = member.inner_callee;
  const auto& call = member.inner_call;
  INTERNAL_CHECK(group != nullptr && callee != nullptr && call != nullptr)
      << "Internal error: incomplete Group member while normalizing its shared ABI";
  CHECK_SPAN(call->args_.size() == callee->params_.size(), call->span_)
      << "Group '" << group->name_ << "' calls member '" << callee->name_ << "' with " << call->args_.size()
      << " arguments, but its signature has " << callee->params_.size();

  std::vector<VarPtr> adapter_params;
  adapter_params.reserve(group->params_.size());
  for (const auto& param : group->params_) {
    auto fresh = std::make_shared<Var>(param->name_hint_, param->GetType(), param->span_);
    adapter_params.push_back(fresh);
  }
  GroupAbiArgSubstituter substitute_arg(canonical_arg_indices, adapter_params);
  std::unordered_set<const Var*> parameter_set;
  for (const auto& param : adapter_params) parameter_set.insert(param.get());

  // Resolve all actuals onto the fresh canonical parameters before inferring
  // type bindings, so their dimensions cannot capture the original Group ABI.
  std::vector<ExprPtr> adapter_args;
  adapter_args.reserve(callee->params_.size());
  for (size_t i = 0; i < callee->params_.size(); ++i) {
    auto argument = substitute_arg.VisitExpr(call->args_[i]);
    var_collectors::VarDefUseCollector uses;
    uses.VisitExpr(argument);
    for (const auto* var : uses.var_uses_ordered) {
      if (parameter_set.count(var)) continue;
      if (auto output = outputs.Find(call->args_[i])) {
        CHECK_SPAN(false, call->span_)
            << "Group '" << group->name_ << "' member '" << callee->name_ << "' argument " << i
            << " consumes output " << output->position << " of producer '" << output->callee->name_
            << "' without a caller-visible output binding. Materialize Tensor results with "
               "ConvertTensorToTileOps or declare an explicit Out parameter; device-computed scalar "
               "results require an explicit tensor output.";
      }
      CHECK_SPAN(false, call->span_) << "Cannot bind Group '" << group->name_ << "' member '" << callee->name_
                                     << "' argument " << i << ": Group-local value '" << var->name_hint_
                                     << "' has no output or parameter binding";
    }
    adapter_args.push_back(std::move(argument));
  }
  // Parameter substitution alone leaves member-local dimensions in aliases,
  // loop state, and return types. Seed DeepClone with the same type bindings
  // used by inline expansion, then give explicit parameter values precedence
  // when a scalar parameter also appears in shaped type metadata.
  auto member_to_adapter = DeduceCallTypeBindings(callee->params_, adapter_args);
  for (size_t i = 0; i < callee->params_.size(); ++i) {
    member_to_adapter[callee->params_[i].get()] = adapter_args[i];
  }
  auto cloned = DeepClone(callee->body_, member_to_adapter);
  StmtPtr adapter_body = cloned.cloned_body;
  if (!peer_renames.empty()) {
    adapter_body = PeerFuncRewriter(peer_renames).VisitStmt(adapter_body);
  }
  // Match DeepClone's one-step seed substitution: actual extent expressions
  // belong to the caller, so N -> N + 32 must not expand N a second time.
  CallShapeInstantiation substitute_member(member_to_adapter);
  std::vector<TypePtr> return_types;
  return_types.reserve(callee->return_types_.size());
  for (const auto& type : callee->return_types_) {
    return_types.push_back(substitute_member.InstantiateType(type));
  }

  return std::make_shared<Function>(adapter_name, std::move(adapter_params), group->param_directions_,
                                    std::move(return_types), std::move(adapter_body), callee->span_,
                                    callee->func_type_, callee->level_, callee->role_, callee->attrs_,
                                    callee->requires_runtime_binding_, callee->ir_stage_);
}

class GroupMemberCallRewriter : public IRMutator {
 public:
  GroupMemberCallRewriter(std::unordered_map<std::string, FunctionPtr> callees,
                          std::vector<VarPtr> canonical_args, GroupAbiArgIndices canonical_arg_indices)
      : callees_(std::move(callees)),
        canonical_args_(std::move(canonical_args)),
        canonical_arg_indices_(std::move(canonical_arg_indices)) {}

  static void ValidateMemberAttrs(const WrapperCallInfo& member,
                                  const GroupAbiArgIndices& canonical_arg_indices) {
    const auto& call = member.inner_call;
    RemapPositionalAttrs(call->attrs_, call->args_, call->op_, call->span_, canonical_arg_indices);
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto updated = As<AssignStmt>(IRMutator::VisitStmt_(op));
    if (updated->value_.get() == op->value_.get() || !As<Var>(updated->var_)) return updated;
    auto type = WithCarriedMemRef(GetAuthoritativeAssignmentType(updated->var_->GetType(), updated->value_),
                                  updated->var_->GetType());
    if (structural_equal(type, updated->var_->GetType())) return updated;
    // A rewritten call can refine its result extent. Retype its SSA receiver
    // and subsequent aliases/projections as well as the expression itself.
    auto fresh = std::make_shared<Var>(updated->var_->name_hint_, type, updated->var_->span_);
    var_remap_[op->var_.get()] = fresh;
    auto result = MutableCopy(updated);
    result->var_ = std::move(fresh);
    return result;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto gv = As<GlobalVar>(op->op_);
    auto it = gv ? callees_.find(gv->name_) : callees_.end();
    if (it == callees_.end()) return IRMutator::VisitExpr_(op);
    std::vector<ExprPtr> args(canonical_args_.begin(), canonical_args_.end());
    auto type = InstantiateResultType(it->second, op->GetType(), false);
    return std::make_shared<Call>(
        std::make_shared<GlobalVar>(it->second->name_), std::move(args), op->kwargs_,
        RemapPositionalAttrs(op->attrs_, op->args_, op->op_, op->span_, canonical_arg_indices_), type,
        op->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto gv = As<GlobalVar>(op->op_);
    auto it = gv ? callees_.find(gv->name_) : callees_.end();
    if (it == callees_.end()) return IRMutator::VisitExpr_(op);
    std::vector<ExprPtr> args(canonical_args_.begin(), canonical_args_.end());
    auto type = InstantiateResultType(it->second, op->GetType(), true);
    return std::make_shared<Submit>(
        std::make_shared<GlobalVar>(it->second->name_), std::move(args), op->deps_, op->kwargs_,
        RemapPositionalAttrs(op->attrs_, op->args_, op->op_, op->span_, canonical_arg_indices_), type,
        op->span_, op->core_num_, op->sync_start_, op->allow_early_resolve_, op->predicate_);
  }

 private:
  TypePtr InstantiateResultType(const FunctionPtr& callee, const TypePtr& original, bool submit) const {
    // Preserve the existing void/legacy inferred-return contract when there is
    // no declared result. A nonempty result must follow the normalized callee,
    // rather than retain a type inferred from the member's old signature.
    if (callee->return_types_.empty()) return original;
    CallShapeMap bindings;
    for (size_t i = 0; i < canonical_args_.size(); ++i) {
      bindings.emplace(callee->params_[i].get(), canonical_args_[i]);
    }
    CallShapeInstantiation instantiate(std::move(bindings));
    std::vector<TypePtr> types;
    for (const auto& type : callee->return_types_) types.push_back(instantiate.InstantiateType(type));
    if (!submit && types.size() == 1) return types.front();
    if (submit) {
      auto tuple = As<TupleType>(original);
      INTERNAL_CHECK_SPAN(tuple && tuple->types_.size() == types.size() + 1, callee->span_)
          << "Internal error: Group member Submit result must retain its trailing TaskId";
      types.push_back(tuple->types_.back());
    }
    return std::make_shared<TupleType>(std::move(types));
  }

  static std::vector<std::pair<std::string, std::any>> RemapPositionalAttrs(
      const std::vector<std::pair<std::string, std::any>>& source_attrs,
      const std::vector<ExprPtr>& source_args, const OpPtr& source_op, const Span& source_span,
      const GroupAbiArgIndices& canonical_arg_indices) {
    std::vector<std::pair<std::string, std::any>> attrs;
    attrs.reserve(source_attrs.size());
    for (const auto& [key, value] : source_attrs) {
      // These directions describe the old member argument list. The standard
      // pipeline derives them again after ExpandMixedKernel, so retaining the
      // vector would make the widened Call/Submit fail constructor validation.
      if (key == kAttrArgDirections) continue;
      if (key != kAttrArgDirectionOverrides) {
        attrs.emplace_back(key, value);
        continue;
      }

      const auto* old_indices = std::any_cast<std::vector<int32_t>>(&value);
      INTERNAL_CHECK_SPAN(old_indices != nullptr, source_span)
          << "Internal error: " << kAttrArgDirectionOverrides << " attr must hold std::vector<int32_t>";
      std::vector<int32_t> remapped_indices;
      remapped_indices.reserve(old_indices->size());
      for (int32_t old_index : *old_indices) {
        INTERNAL_CHECK_SPAN(old_index >= 0 && static_cast<size_t>(old_index) < source_args.size(),
                            source_span)
            << "Internal error: " << kAttrArgDirectionOverrides << " index " << old_index
            << " is out of range for Group member call to '" << source_op->name_ << "'";
        const auto& old_arg = source_args[static_cast<size_t>(old_index)];
        auto canonical_it = canonical_arg_indices.find(old_arg.get());
        CHECK_SPAN(canonical_it != canonical_arg_indices.end(), source_span)
            << "Cannot map pl.no_dep argument " << old_index << " of Group member '" << source_op->name_
            << "' to a canonical Group parameter";
        remapped_indices.push_back(canonical_it->second);
      }
      attrs.emplace_back(key, std::move(remapped_indices));
    }
    return attrs;
  }

  std::unordered_map<std::string, FunctionPtr> callees_;
  std::vector<VarPtr> canonical_args_;
  GroupAbiArgIndices canonical_arg_indices_;
};

std::string ReserveAdapterName(const std::string& base, std::unordered_set<std::string>& used_names) {
  if (used_names.insert(base).second) return base;
  for (size_t suffix = 1;; ++suffix) {
    std::string candidate = base + "_" + std::to_string(suffix);
    if (used_names.insert(candidate).second) return candidate;
  }
}

}  // namespace

NormalizedGroups NormalizeHandWrittenGroupAbis(const ProgramPtr& program,
                                               const std::vector<FunctionPtr>& functions) {
  // Every Group queries the same immutable program. Reused members need one
  // callee analysis for this whole normalization, not one per private adapter.
  return_lineage::ScopedTraceCache lineage_cache;
  KernelCallCounter call_counter;
  for (const auto& func : functions) {
    if (func && func->body_) call_counter.VisitStmt(func->body_);
  }

  std::unordered_set<std::string> used_names;
  std::unordered_map<std::string, FunctionPtr> replacements;
  std::vector<FunctionPtr> adapters;
  CalleeOutputSummaries output_summaries;
  for (const auto& func : functions) used_names.insert(func->name_);

  for (const auto& group : functions) {
    if (!group || group->func_type_ != FunctionType::Group) continue;

    WrapperCallInfo aic;
    WrapperCallInfo aiv;
    for (const auto& member : CollectInnerCalls(group, program)) {
      if (member.inner_callee->func_type_ == FunctionType::AIC && !aic.inner_call) {
        aic = member;
      } else if (member.inner_callee->func_type_ == FunctionType::AIV && !aiv.inner_call) {
        aiv = member;
      }
    }
    // AIV-only Groups and pre-expansion Groups are handled by their existing
    // dispatch paths. A mixed Group needs exactly the shared AIC/AIV ABI here.
    if (!aic.inner_call || !aiv.inner_call) continue;
    const bool needs_abi_normalization =
        !IsCanonicalGroupMemberCall(aic.inner_call, aic.inner_callee, group) ||
        !IsCanonicalGroupMemberCall(aiv.inner_call, aiv.inner_callee, group);
    const bool needs_dual_aiv_dispatch = NeedsInferredNoSplitDualAivDispatch(aiv.inner_callee);
    if (!needs_abi_normalization && !needs_dual_aiv_dispatch) continue;

    if (needs_abi_normalization) {
      CHECK_SPAN(!aic.inner_callee->HasAttr(kAttrExternalSource) &&
                     !aiv.inner_callee->HasAttr(kAttrExternalSource) &&
                     !aic.inner_callee->requires_runtime_binding_ &&
                     !aiv.inner_callee->requires_runtime_binding_,
                 group->span_)
          << "Mixed Group '" << group->name_
          << "' has AIC/AIV members with different argument layouts. External or runtime-bound members "
             "cannot be adapted; declare both members with the same signature and forward the Group's full "
             "parameter list.";
    }

    const bool exclusive_pair =
        call_counter.Count(aic.inner_callee->name_) == 1 && call_counter.Count(aiv.inner_callee->name_) == 1;
    const std::string aic_name = exclusive_pair
                                     ? aic.inner_callee->name_
                                     : ReserveAdapterName(group->name_ + "_aic_adapter", used_names);
    const std::string aiv_name = exclusive_pair
                                     ? aiv.inner_callee->name_
                                     : ReserveAdapterName(group->name_ + "_aiv_adapter", used_names);
    const std::unordered_map<std::string, std::string> peer_renames = {{aic.inner_callee->name_, aic_name},
                                                                       {aiv.inner_callee->name_, aiv_name}};

    // First retain every producer return position, including non-parameter
    // results. Only then resolve each consumer argument to the storage supplied
    // at that call site. Tensor lowering has already made computed results
    // explicit Out parameters; missing carriers are diagnosed before cloning.
    GroupOutputBindings output_bindings(group, program, output_summaries);
    auto canonical_arg_indices = BuildGroupAbiArgIndices(group, aic, aiv, program, output_bindings);
    // Keep call-site metadata diagnostics ahead of body cloning: an invalid
    // no_dep argument must name its canonical-parameter binding failure before
    // a member adapter can report the same value as a missing output carrier.
    GroupMemberCallRewriter::ValidateMemberAttrs(aic, canonical_arg_indices);
    GroupMemberCallRewriter::ValidateMemberAttrs(aiv, canonical_arg_indices);
    auto normalized_aic =
        BuildGroupAbiAdapter(group, aic, aic_name, peer_renames, canonical_arg_indices, output_bindings);
    auto normalized_aiv =
        BuildGroupAbiAdapter(group, aiv, aiv_name, peer_renames, canonical_arg_indices, output_bindings);
    // Normalize member signatures before rebuilding their call result types.
    std::unordered_map<std::string, FunctionPtr> callees = {{aic.inner_callee->name_, normalized_aic},
                                                            {aiv.inner_callee->name_, normalized_aiv}};
    auto mutable_group = MutableCopy(group);
    mutable_group->body_ = GroupMemberCallRewriter(std::move(callees), group->params_, canonical_arg_indices)
                               .VisitStmt(group->body_);
    if (needs_dual_aiv_dispatch) normalized_aiv = WithDualAivDispatch(normalized_aiv);
    if (exclusive_pair) {
      replacements[aic_name] = normalized_aic;
      replacements[aiv_name] = normalized_aiv;
    } else {
      adapters.push_back(normalized_aic);
      adapters.push_back(normalized_aiv);
    }

    replacements[group->name_] = mutable_group;
  }

  std::vector<FunctionPtr> result;
  result.reserve(functions.size() + adapters.size());
  result.insert(result.end(), adapters.begin(), adapters.end());
  for (const auto& func : functions) {
    auto it = replacements.find(func->name_);
    result.push_back(it == replacements.end() ? func : it->second);
  }
  return {std::move(result)};
}

}  // namespace expand_mixed_kernel
}  // namespace ir
}  // namespace pypto
