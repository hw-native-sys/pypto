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

#include "pypto/codegen/orchestration/iter_arg_carry_analyzer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

namespace {

/// Find a ForStmt within ``body`` whose ``return_vars_`` contains ``target``.
/// Returns nullptr if none. Used to chase Sequential→Parallel array threading.
ForStmtPtr FindForStmtByReturnVar(const StmtPtr& body, const Var* target) {
  class Finder : public IRVisitor {
   public:
    ForStmtPtr result;
    const Var* target = nullptr;
    void VisitStmt_(const ForStmtPtr& f) override {
      if (result) return;
      for (const auto& rv : f->return_vars_) {
        if (rv.get() == target) {
          result = f;
          return;
        }
      }
      IRVisitor::VisitStmt_(f);
    }
  };
  Finder finder;
  finder.target = target;
  finder.VisitStmt(body);
  return finder.result;
}

std::vector<std::unordered_set<const Var*>> ComputeAliasClasses(const ForStmtPtr& for_stmt,
                                                                const BodyAliases& body_aliases,
                                                                const ProgramPtr& program) {
  std::vector<std::unordered_set<const Var*>> aliases(for_stmt->iter_args_.size());
  for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
    aliases[i].insert(for_stmt->iter_args_[i].get());
  }

  std::unordered_map<const Var*, AssignStmtPtr> var_to_assign;
  for (const auto& a : body_aliases.assigns) {
    var_to_assign[a->var_.get()] = a;
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& assign : body_aliases.assigns) {
      // TupleGetItemExpr: climb to the tuple-producing call or submit and
      // resolve the corresponding output arg (Submit is viewed as a Call
      // via AsCallOrSubmitView, so its output args alias identically).
      if (auto tge = As<TupleGetItemExpr>(assign->value_)) {
        auto tuple_var = AsVarLike(tge->tuple_);
        if (tuple_var) {
          auto it = var_to_assign.find(tuple_var.get());
          if (it != var_to_assign.end()) {
            auto tcall = AsCallOrSubmitView(it->second->value_);
            if (tcall) {
              auto tdirs = tcall->GetArgDirections();
              if (tdirs.size() == tcall->args_.size()) {
                int64_t out_seen = 0;
                int64_t target_idx = static_cast<int64_t>(tge->index_);
                for (size_t a = 0; a < tdirs.size(); ++a) {
                  if (tdirs[a] != ArgDirection::OutputExisting && tdirs[a] != ArgDirection::InOut &&
                      tdirs[a] != ArgDirection::Output) {
                    continue;
                  }
                  if (out_seen == target_idx) {
                    auto out_arg = AsVarLike(tcall->args_[a]);
                    if (out_arg) {
                      for (auto& cls : aliases) {
                        if (cls.count(out_arg.get()) && !cls.count(assign->var_.get())) {
                          cls.insert(assign->var_.get());
                          changed = true;
                        }
                      }
                    }
                    break;
                  }
                  ++out_seen;
                }
              }
            }
          }
        }
        continue;
      }
      auto call = AsCallOrSubmitView(assign->value_);
      if (!call) continue;
      // tensor.assemble: result var aliases its first arg (the target).
      if (call->op_->name_ == "tensor.assemble" && !call->args_.empty()) {
        auto first_arg = AsVarLike(call->args_[0]);
        if (first_arg) {
          for (auto& cls : aliases) {
            if (cls.count(first_arg.get()) && !cls.count(assign->var_.get())) {
              cls.insert(assign->var_.get());
              changed = true;
            }
          }
        }
      }
      // Calls with output_existing/inout args (e.g. InCore kernels):
      // the result aliases the Out/InOut arg the callee actually returns.
      auto call_dirs = call->GetArgDirections();
      if (call_dirs.size() == call->args_.size()) {
        FunctionPtr call_callee = program->GetFunction(call->op_->name_);
        std::optional<size_t> returned_idx = FindReturnedParamIndex(call_callee, program);
        for (size_t a = 0; a < call_dirs.size(); ++a) {
          if (call_dirs[a] != ArgDirection::OutputExisting && call_dirs[a] != ArgDirection::InOut) {
            continue;
          }
          if (returned_idx.has_value() && a != *returned_idx) {
            continue;
          }
          auto out_arg = AsVarLike(call->args_[a]);
          if (!out_arg) continue;
          for (auto& cls : aliases) {
            if (cls.count(out_arg.get()) && !cls.count(assign->var_.get())) {
              cls.insert(assign->var_.get());
              changed = true;
            }
          }
          break;
        }
      }
    }
    // Nested ForStmts: the parent's carry threaded through a nested
    // loop comes out via the nested loop's return_var. ArrayType iter_args
    // are excluded — they own a fresh C-stack array at each level.
    for (const auto& nf : body_aliases.nested_fors) {
      for (size_t k = 0; k < nf->iter_args_.size(); ++k) {
        if (As<ArrayType>(nf->iter_args_[k]->GetType())) continue;
        auto init_var = AsVarLike(nf->iter_args_[k]->initValue_);
        if (!init_var) continue;
        if (k >= nf->return_vars_.size()) continue;
        const auto* rv = nf->return_vars_[k].get();
        for (auto& cls : aliases) {
          if (cls.count(init_var.get()) && !cls.count(rv)) {
            cls.insert(rv);
            changed = true;
          }
        }
      }
    }
  }
  return aliases;
}

}  // namespace

IterArgCarryAnalyzer::IterArgCarryAnalyzer(ProgramPtr program, int manual_scope_depth)
    : program_(std::move(program)), manual_scope_depth_(manual_scope_depth) {}

std::vector<IterArgCarryPlan> IterArgCarryAnalyzer::Analyze(const ForStmtPtr& for_stmt) {
  std::vector<IterArgCarryPlan> plans(for_stmt->iter_args_.size());

  auto yield = transform_utils::GetLastYieldStmt(UnwrapAutoScope(for_stmt->body_));
  if (yield) {
    INTERNAL_CHECK_SPAN(yield->value_.size() == for_stmt->iter_args_.size(), for_stmt->span_)
        << "Internal error: ForStmt yield/iter_args size mismatch";

    auto body_aliases = CollectBodyAliases(for_stmt->body_);
    auto aliases = ComputeAliasClasses(for_stmt, body_aliases, program_);

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      auto yield_var = AsVarLike(yield->value_[i]);
      plans[i].is_rebind = !yield_var || !aliases[i].count(yield_var.get());
      if (auto sty = As<ScalarType>(for_stmt->iter_args_[i]->GetType())) {
        if (sty->dtype_ == DataType::TASK_ID) plans[i].is_rebind = true;
      }
    }
  }

  if (manual_scope_depth_ > 0) {
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      if (plans[i].is_rebind) {
        plans[i].array_size = ResolveArrayCarrySize(for_stmt, i);
      }
    }
  }

  if (for_stmt->kind_ == ForKind::Parallel && manual_scope_depth_ > 0) {
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      if (!plans[i].is_rebind) continue;
      auto sty = As<ScalarType>(for_stmt->iter_args_[i]->GetType());
      if (!sty || sty->dtype_ != DataType::TASK_ID) continue;
      CHECK(plans[i].array_size > 0) << "manual_scope: pl.parallel loops carrying a manual_scope dep "
                                     << "(via ``deps=[...]``) must have a statically-known trip count. "
                                     << "The runtime fence requires a PTO2TaskId[N] array of fixed N. "
                                     << "Either make the parallel loop's trip count a Python int "
                                     << "(e.g. ``pl.parallel(4)``) or restructure to put the parallel "
                                     << "loop inside a const-bounded scope.";
    }
  }

  return plans;
}

int64_t IterArgCarryAnalyzer::ResolveArrayCarrySize(const ForStmtPtr& for_stmt, size_t idx) const {
  if (idx >= for_stmt->iter_args_.size()) return 0;
  const auto& iter_arg = for_stmt->iter_args_[idx];
  auto sty = As<ScalarType>(iter_arg->GetType());
  if (!sty || sty->dtype_ != DataType::TASK_ID) return 0;
  if (for_stmt->kind_ == ForKind::Parallel) {
    return EvalConstTripCount(for_stmt);
  }
  if (for_stmt->kind_ != ForKind::Sequential) return 0;
  auto yield = transform_utils::GetLastYieldStmt(UnwrapAutoScope(for_stmt->body_));
  if (!yield || idx >= yield->value_.size()) return 0;
  auto yield_var = AsVarLike(yield->value_[idx]);
  if (!yield_var) return 0;
  // Search the raw body (not unwrapped): FindForStmtByReturnVar is a visitor
  // that descends through RuntimeScopeStmt nodes, unlike GetLastYieldStmt.
  auto inner = FindForStmtByReturnVar(for_stmt->body_, yield_var.get());
  if (!inner) return 0;
  size_t inner_idx = SIZE_MAX;
  for (size_t j = 0; j < inner->return_vars_.size(); ++j) {
    if (inner->return_vars_[j].get() == yield_var.get()) {
      inner_idx = j;
      break;
    }
  }
  if (inner_idx == SIZE_MAX) return 0;
  return ResolveArrayCarrySize(inner, inner_idx);
}

}  // namespace codegen
}  // namespace pypto
