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
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {
namespace {

using op_predicates::IsBuiltinOp;

}  // namespace

BufferRootCollector::BufferRootCollector(ProgramPtr program, AmbiguousRootPolicy ambiguous_policy)
    : program_(std::move(program)), ambiguous_policy_(ambiguous_policy) {}

void BufferRootCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    RecordRootCandidates(param, {param.get()});
  }
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
    auto roots = ResolveRootCandidates(then_yield->value_[i]);
    auto else_roots = ResolveRootCandidates(else_yield->value_[i]);
    for (const Var* root : else_roots) {
      if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
    }
    const bool unresolved = roots.empty() || IsAmbiguous(then_yield->value_[i]) ||
                            IsAmbiguous(else_yield->value_[i]) ||
                            ResolveRootCandidates(then_yield->value_[i]).empty() || else_roots.empty();
    RecordRootCandidates(if_stmt->return_vars_[i], roots, unresolved);
  }
}

void BufferRootCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  auto init_roots = InitializeLoopCarryRoots(for_stmt->iter_args_);
  IRVisitor::VisitStmt_(for_stmt);
  RecordLoopReturnRoots(for_stmt->body_, for_stmt->return_vars_, init_roots,
                        transform_utils::EvalConstTripCount(for_stmt).value_or(0) > 0);
}

void BufferRootCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  auto init_roots = InitializeLoopCarryRoots(while_stmt->iter_args_);
  IRVisitor::VisitStmt_(while_stmt);
  // A while condition may be false before the first iteration, so both the
  // initial and yielded roots remain possible.
  RecordLoopReturnRoots(while_stmt->body_, while_stmt->return_vars_, init_roots,
                        /*guaranteed_to_run=*/false);
}

std::vector<BufferRootCollector::RootCandidates> BufferRootCollector::InitializeLoopCarryRoots(
    const std::vector<IterArgPtr>& iter_args) {
  std::vector<RootCandidates> init_roots;
  init_roots.reserve(iter_args.size());
  for (const auto& iter_arg : iter_args) {
    auto roots = ResolveRootCandidates(iter_arg->initValue_);
    init_roots.push_back(roots);
    RecordRootCandidates(iter_arg, roots, IsAmbiguous(iter_arg->initValue_));
  }
  return init_roots;
}

void BufferRootCollector::RecordLoopReturnRoots(const StmtPtr& body, const std::vector<VarPtr>& return_vars,
                                                const std::vector<RootCandidates>& init_roots,
                                                bool guaranteed_to_run) {
  // Resolve after visiting the body so fresh-rebind assignments already have
  // roots. An in-place yield of the IterArg still resolves to its init root.
  auto yield = transform_utils::GetLastYieldStmt(body);
  if (!yield) return;

  for (size_t i = 0; i < return_vars.size() && i < yield->value_.size(); ++i) {
    auto roots = ResolveRootCandidates(yield->value_[i]);
    bool unresolved = roots.empty() || IsAmbiguous(yield->value_[i]);
    if (!guaranteed_to_run && i < init_roots.size()) {
      if (init_roots[i].empty()) unresolved = true;
      for (const Var* root : init_roots[i]) {
        if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
      }
    }
    RecordRootCandidates(return_vars[i], roots, unresolved);
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
    if (IsOp(call, "tensor.create") || IsOp(call, "tensor.slice")) {
      RecordRootCandidates(assign->var_, {assign->var_.get()});
    } else if (IsOp(call, "tensor.assemble")) {
      if (call->args_.size() == 3) {
        RecordRootCandidates(assign->var_, ResolveRootCandidates(call->args_[0]),
                             IsAmbiguous(call->args_[0]));
      }
    } else if (!IsBuiltinOp(op_name)) {
      auto out_roots = CollectCallOutputRoots(call);
      if (As<TupleType>(call->GetType())) {
        std::vector<RootInfo> roots;
        roots.reserve(out_roots.size());
        for (const auto& entry : out_roots) roots.push_back(entry.info);
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      } else {
        auto info = SelectReturnRootInfo(out_roots, call->GetType());
        if (!info.roots.empty()) {
          RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
        }
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_output_roots_.find(tuple_var.get());
      if (it != tuple_output_roots_.end() && tuple_get->index_ < static_cast<int>(it->second.size()) &&
          !it->second[tuple_get->index_].roots.empty()) {
        const auto& info = it->second[tuple_get->index_];
        RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
      }
    }
  } else if (AsVarLike(assign->value_)) {
    RecordRootCandidates(assign->var_, ResolveRootCandidates(assign->value_), IsAmbiguous(assign->value_));
  }
  IRVisitor::VisitStmt_(assign);
}

BufferRootCollector::RootCandidates BufferRootCollector::ResolveRootCandidates(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  if (!var) return {};
  auto it = root_candidates_.find(var.get());
  return it != root_candidates_.end() ? it->second : RootCandidates{};
}

bool BufferRootCollector::IsAmbiguous(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  return var && ambiguous_buffer_vars.count(var.get()) != 0;
}

void BufferRootCollector::RecordRootCandidates(const VarPtr& var, const RootCandidates& roots,
                                               bool unresolved) {
  if (!var) return;
  RootCandidates unique;
  unique.reserve(roots.size());
  for (const Var* root : roots) {
    if (root && std::find(unique.begin(), unique.end(), root) == unique.end()) unique.push_back(root);
  }
  if (unique.empty()) return;

  root_candidates_[var.get()] = unique;
  const bool ambiguous = unresolved || unique.size() > 1;
  if (ambiguous) {
    ambiguous_buffer_vars.insert(var.get());
    if (ambiguous_policy_ == AmbiguousRootPolicy::kFirstOutput) {
      buffer_roots[var.get()] = unique.front();
    } else {
      buffer_roots.erase(var.get());
    }
  } else {
    ambiguous_buffer_vars.erase(var.get());
    buffer_roots[var.get()] = unique.front();
  }
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
    roots.push_back(OutputRoot{RootInfo{ResolveRootCandidates(call->args_[i]), IsAmbiguous(call->args_[i])},
                               call->args_[i]->GetType()});
  }
  return roots;
}

BufferRootCollector::RootInfo BufferRootCollector::SelectReturnRootInfo(
    const std::vector<OutputRoot>& out_roots, const TypePtr& return_type) const {
  if (out_roots.empty()) return RootInfo{{}, false};
  if (out_roots.size() == 1) return out_roots[0].info;

  const OutputRoot* match = nullptr;
  bool ambiguous = false;
  for (const auto& candidate : out_roots) {
    if (!candidate.info.roots.empty() && TypesMatchShapeDtype(candidate.type, return_type)) {
      if (match == nullptr) {
        match = &candidate;
      } else if (match->info.roots != candidate.info.roots ||
                 match->info.ambiguous != candidate.info.ambiguous) {
        ambiguous = true;
      }
    }
  }
  if (match && !ambiguous) return match->info;
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
    return out_roots[0].info;
  }
  return RootInfo{{}, false};
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
