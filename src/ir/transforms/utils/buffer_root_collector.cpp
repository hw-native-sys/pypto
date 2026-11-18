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

#include "pypto/ir/transforms/utils/buffer_root_collector.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/hash.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/vf_helpers.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {
namespace {

using op_predicates::IsBuiltinOp;

}  // namespace

BufferRootCollector::BufferRootCollector(ProgramPtr program, AmbiguousRootPolicy ambiguous_policy,
                                         bool follow_reshape)
    : program_(std::move(program)), ambiguous_policy_(ambiguous_policy), follow_reshape_(follow_reshape) {}

void BufferRootCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    RecordOwningRoot(param);
  }
}

void BufferRootCollector::VisitStmt(const StmtPtr& stmt) {
  // Nested visits share the cache; the outermost visit clears it before a
  // caller can mutate the Program or reuse this analysis for another one.
  return_lineage::ScopedTraceCache lineage_cache;
  IRVisitor::VisitStmt(stmt);
}

void BufferRootCollector::VisitStmt_(const IfStmtPtr& if_stmt) {
  IRVisitor::VisitStmt_(if_stmt);
  if (if_stmt->return_vars_.empty() || !if_stmt->else_body_.has_value()) return;

  auto then_yield = transform_utils::GetLastYieldStmt(if_stmt->then_body_);
  auto else_yield = transform_utils::GetLastYieldStmt(*if_stmt->else_body_);
  if (!then_yield || !else_yield) return;

  for (size_t i = 0;
       i < if_stmt->return_vars_.size() && i < then_yield->value_.size() && i < else_yield->value_.size();
       ++i) {
    const auto& then_value = then_yield->value_[i];
    const auto& else_value = else_yield->value_[i];
    const auto& then_roots = ResolveRootCandidates(then_value);
    const auto& else_roots = ResolveRootCandidates(else_value);
    const bool then_ambiguous = IsAmbiguous(then_value);
    const bool else_ambiguous = IsAmbiguous(else_value);
    const bool unresolved = then_roots.set == 0 || else_roots.set == 0 || then_ambiguous || else_ambiguous;
    MergeRootCandidates(if_stmt->return_vars_[i], {then_roots, else_roots}, unresolved);
  }
}

void BufferRootCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  const auto trip_count = transform_utils::EvalConstTripCount(for_stmt);
  InitializeLoopCarryRoots(for_stmt->iter_args_, for_stmt->body_, trip_count.value_or(2) > 1);
  IRVisitor::VisitStmt_(for_stmt);
  RecordLoopReturnRoots(for_stmt->body_, for_stmt->return_vars_, for_stmt->iter_args_,
                        trip_count.value_or(0) > 0);
}

void BufferRootCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  InitializeLoopCarryRoots(while_stmt->iter_args_, while_stmt->body_, /*may_repeat=*/true);
  IRVisitor::VisitStmt_(while_stmt);
  // A while condition may be false before the first iteration, so both the
  // initial and yielded roots remain possible.
  RecordLoopReturnRoots(while_stmt->body_, while_stmt->return_vars_, while_stmt->iter_args_,
                        /*guaranteed_to_run=*/false);
}

void BufferRootCollector::InitializeLoopCarryRoots(const std::vector<IterArgPtr>& iter_args,
                                                   const StmtPtr& body, bool may_repeat) {
  const auto yield =
      follow_reshape_ && may_repeat && !iter_args.empty() ? transform_utils::GetLastYieldStmt(body) : nullptr;
  for (size_t i = 0; i < iter_args.size(); ++i) {
    const auto& iter_arg = iter_args[i];
    const auto& init = iter_arg->initValue_;
    if (!yield || i >= yield->value_.size() || yield->value_[i].get() == iter_arg.get()) {
      ShareRootCandidates(iter_arg, init);
      continue;
    }

    // An external yield can already be resolved before visiting the body.
    // For an unresolved body value, the initializer alone does not prove
    // that later iterations refer to the same storage.
    const auto& next = yield->value_[i];
    const auto& initial_roots = ResolveRootCandidates(init);
    const auto& next_roots = ResolveRootCandidates(next);
    const bool unresolved =
        initial_roots.set == 0 || next_roots.set == 0 || IsAmbiguous(init) || IsAmbiguous(next);
    MergeRootCandidates(iter_arg, {initial_roots, next_roots}, unresolved);
  }
}

void BufferRootCollector::RecordLoopReturnRoots(const StmtPtr& body, const std::vector<VarPtr>& return_vars,
                                                const std::vector<IterArgPtr>& iter_args,
                                                bool guaranteed_to_run) {
  // Resolve after visiting the body so fresh-rebind assignments already have
  // roots. An in-place yield of the IterArg still resolves to its init root.
  auto yield = transform_utils::GetLastYieldStmt(body);
  if (!yield) return;

  for (size_t i = 0; i < return_vars.size() && i < yield->value_.size(); ++i) {
    const auto& value = yield->value_[i];
    const auto& yielded_roots = ResolveRootCandidates(value);
    bool unresolved = yielded_roots.set == 0 || IsAmbiguous(value);
    if (!guaranteed_to_run && i < iter_args.size()) {
      const auto& init_roots = ResolveRootCandidates(iter_args[i]);
      unresolved = unresolved || init_roots.set == 0 || (follow_reshape_ && IsAmbiguous(iter_args[i]));
      MergeRootCandidates(return_vars[i], {yielded_roots, init_roots}, unresolved);
    } else {
      BindRootCandidates(return_vars[i], yielded_roots, unresolved);
    }
  }
}

void BufferRootCollector::VisitStmt_(const AssignStmtPtr& assign) {
  // Submit (pl.submit inside a pl.manual_scope) is a sibling call-like kind;
  // route it through the Call-shaped view so a task launch's Out/InOut roots are
  // tracked identically to a plain Call (results keyed on the stable binding
  // Var). The view preserves args_ and the TASK_ID-augmented return type, so the
  // tuple path maps the submit-result projections to the callee's Out roots and
  // leaves the trailing TASK_ID element unmapped
  // (see .claude/rules/pass-submit-awareness.md).
  if (auto call = transform_utils::AsCallOrSubmitView(assign->value_)) {
    const std::string& op_name = call->op_->name_;
    if (auto helper = ResolveVFCall(call, program_)) {
      auto aliases = VFReturnedArguments(helper, call);
      if (As<TupleType>(call->GetType())) {
        std::vector<OutputRoot> roots;
        roots.reserve(aliases.size());
        for (const auto& alias : aliases) {
          roots.push_back({alias, ResolveRootCandidates(alias), IsAmbiguous(alias)});
        }
        tuple_output_roots_[assign->var_.get()] =
            std::make_shared<const std::vector<OutputRoot>>(std::move(roots));
      } else if (!aliases.empty()) {
        ShareRootCandidates(assign->var_, aliases[0]);
      }
    } else if (follow_reshape_ && IsOp(call, "tensor.reshape") && !call->args_.empty()) {
      ShareRootCandidates(assign->var_, call->args_[0]);
    } else if (IsOp(call, "tensor.create") || IsOp(call, "tensor.slice")) {
      RecordOwningRoot(assign->var_);
    } else if (IsOp(call, "tensor.assemble")) {
      if (call->args_.size() == 3) {
        ShareRootCandidates(assign->var_, call->args_[0]);
      }
    } else if (!IsBuiltinOp(op_name)) {
      auto out_roots = CollectCallOutputRoots(call);
      if (As<TupleType>(call->GetType())) {
        tuple_output_roots_[assign->var_.get()] =
            std::make_shared<const std::vector<OutputRoot>>(std::move(out_roots));
      } else {
        RecordCallReturnRoots(assign->var_, out_roots, call);
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_output_roots_.find(tuple_var.get());
      if (it != tuple_output_roots_.end() && tuple_get->index_ >= 0 &&
          tuple_get->index_ < static_cast<int>(it->second->size())) {
        const auto& info = (*it->second)[tuple_get->index_];
        BindRootCandidates(assign->var_, info.roots, info.ambiguous);
      }
    }
  } else if (auto source = AsVarLike(assign->value_)) {
    if (auto it = tuple_output_roots_.find(source.get()); it != tuple_output_roots_.end()) {
      tuple_output_roots_[assign->var_.get()] = it->second;
    }
    ShareRootCandidates(assign->var_, source);
  }
  IRVisitor::VisitStmt_(assign);
}

void BufferRootCollector::VisitStmt_(const SimtLaunchStmtPtr& launch) {
  // GM results are new dependency versions of the exact same reference. Tile
  // results have value semantics and are materialized separately before planning.
  for (size_t i = 0; i < launch->results_.size(); ++i) {
    if (!AsTensorTypeLike(launch->results_[i]->GetType())) continue;
    const auto& argument = launch->args_[launch->result_param_indices_[i]];
    ShareRootCandidates(launch->results_[i], argument);
  }
  IRVisitor::VisitStmt_(launch);
}

const BufferRootCollector::RootCandidates& BufferRootCollector::ResolveRootCandidates(
    const ExprPtr& expr) const {
  static const RootCandidates kNone;
  const auto root = root_candidates_.find(AsVarLike(expr).get());
  return root == root_candidates_.end() ? kNone : root->second;
}

void BufferRootCollector::ShareRootCandidates(const VarPtr& var, const ExprPtr& source) {
  BindRootCandidates(var, ResolveRootCandidates(source), IsAmbiguous(source));
}

bool BufferRootCollector::IsAmbiguous(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  return var && ambiguous_buffer_vars.count(var.get()) != 0;
}

void BufferRootCollector::RecordOwningRoot(const VarPtr& var) {
  if (!var) return;
  const auto [entry, inserted] = owning_sets_.try_emplace(var.get(), root_sets_.size());
  if (inserted) root_sets_.push_back({var.get(), 0, 0});
  BindRootCandidates(var, {entry->second, var.get()}, false);
}

void BufferRootCollector::BindRootCandidates(const VarPtr& var, const RootCandidates& roots,
                                             bool unresolved) {
  if (!var || roots.set == 0) return;
  root_candidates_[var.get()] = roots;
  if (unresolved || root_sets_[roots.set].root == nullptr) {
    ambiguous_buffer_vars.insert(var.get());
    if (ambiguous_policy_ == AmbiguousRootPolicy::kFirstOutput) {
      buffer_roots[var.get()] = roots.first;
    } else {
      buffer_roots.erase(var.get());
    }
  } else {
    ambiguous_buffer_vars.erase(var.get());
    buffer_roots[var.get()] = roots.first;
  }
}

void BufferRootCollector::MergeRootCandidates(const VarPtr& var, const std::vector<RootCandidates>& inputs,
                                              bool unresolved) {
  RootCandidates merged;
  for (const auto& input : inputs) {
    if (input.set == 0) continue;
    if (merged.set == 0) merged.first = input.first;
    merged.set = UnionRootSets(merged.set, input.set);
  }
  BindRootCandidates(var, merged, unresolved);
}

BufferRootCollector::RootSetId BufferRootCollector::RootSetOf(const Var* var) const {
  const auto it = root_candidates_.find(var);
  return it == root_candidates_.end() ? 0 : it->second.set;
}

size_t BufferRootCollector::RootSetPairHash::operator()(const RootSetPair& pair) const {
  return hash_combine(pair.first, pair.second);
}

BufferRootCollector::RootSetId BufferRootCollector::UnionRootSets(RootSetId left, RootSetId right) {
  if (left == 0 || left == right) return right;
  if (right == 0) return left;
  if (left > right) std::swap(left, right);
  const auto [entry, inserted] = interned_sets_.try_emplace(RootSetPair{left, right}, root_sets_.size());
  if (inserted) root_sets_.push_back({nullptr, left, right});
  return entry->second;
}

std::vector<BufferRootCollector::OutputRoot> BufferRootCollector::CollectCallOutputRoots(
    const CallPtr& call) const {
  auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return {};

  std::vector<OutputRoot> roots;
  for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
    if (callee->param_directions_[i] != ParamDirection::Out &&
        callee->param_directions_[i] != ParamDirection::InOut) {
      continue;
    }
    const auto& arg = call->args_[i];
    roots.push_back({arg, ResolveRootCandidates(arg), IsAmbiguous(arg)});
  }
  return roots;
}

void BufferRootCollector::RecordCallReturnRoots(const VarPtr& result,
                                                const std::vector<OutputRoot>& out_roots,
                                                const CallPtr& call) {
  if (out_roots.empty()) return;
  const auto record_output = [&](const OutputRoot& output) {
    BindRootCandidates(result, output.roots, output.ambiguous);
  };
  if (out_roots.size() == 1) {
    record_output(out_roots.front());
    return;
  }

  const OutputRoot* match = nullptr;
  bool ambiguous = false;
  for (const auto& candidate : out_roots) {
    if (!TypesMatchShapeDtype(candidate.argument->GetType(), call->GetType())) continue;
    if (candidate.roots.set == 0) {
      // A compatible output with unknown storage could be the result.
      // Its absence from the root map does not prove the other match.
      ambiguous = ambiguous || follow_reshape_;
    } else if (!match) {
      match = &candidate;
    } else if (match->roots.set != candidate.roots.set || match->roots.first != candidate.roots.first ||
               match->ambiguous != candidate.ambiguous) {
      ambiguous = true;
    }
  }
  if (match && !ambiguous) {
    record_output(*match);
    return;
  }

  if (follow_reshape_ && ambiguous) {
    // Resolve a tie between same-shaped outputs from the callee's actual
    // return. Following reshape can expose a second compatible buffer.
    const auto callee = program_->GetFunction(call->op_->name_);
    if (callee && callee->return_types_.size() == 1) {
      const auto param = return_lineage::ReturnedParamIndex(callee, program_);
      if (param && *param < call->args_.size() && callee->param_directions_[*param] != ParamDirection::In) {
        const auto& arg = call->args_[*param];
        // Callee identity is useful only if the actual argument has one
        // proven root. A partial set still needs the compatible candidates.
        if (ResolveRootCandidates(arg).set != 0 && !IsAmbiguous(arg)) {
          ShareRootCandidates(result, arg);
          return;
        }
      }
    }
    // A consumer may read or write whichever compatible buffer was returned.
    // Retain every compatible root when the return proof is unavailable.
    std::vector<RootCandidates> inputs;
    for (const auto& candidate : out_roots) {
      if (TypesMatchShapeDtype(candidate.argument->GetType(), call->GetType())) {
        inputs.push_back(candidate.roots);
      }
    }
    MergeRootCandidates(result, inputs, /*unresolved=*/true);
    return;
  }
  // No provable unambiguous type match (0 matches, or >1 distinct candidates).
  // The fallback depends on what the consumer needs when the owning buffer
  // can't be pinned down:
  //   kSkip        — record no root. Fusion / aliasing is an optimization, so
  //                  skipping it (no root -> no aliasing) is always safe,
  //                  whereas guessing could re-alias a scratch onto the output.
  //   kFirstOutput — fall back to the first Out/InOut root, matching the naive
  //                  pre-dedup behavior. DeriveCallDirections needs *some* root
  //                  so a later write to the returned var still promotes to
  //                  InOut; a null root would silently drop the WAW/InOut dep.
  if (ambiguous_policy_ == AmbiguousRootPolicy::kFirstOutput) {
    record_output(out_roots.front());
  }
}

bool BufferRootCollector::TypesMatchShapeDtype(const TypePtr& a, const TypePtr& b) {
  auto ta = As<TensorType>(a);
  auto tb = As<TensorType>(b);
  if (!ta || !tb) return false;
  if (ta->dtype_ != tb->dtype_) return false;
  if (ta->shape_.size() != tb->shape_.size()) return false;
  for (size_t i = 0; i < ta->shape_.size(); ++i) {
    if (!AreExprsEqual(ta->shape_[i], tb->shape_[i])) return false;
  }
  return true;
}

}  // namespace buffer_root
}  // namespace ir
}  // namespace pypto
