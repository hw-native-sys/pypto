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

#ifndef SRC_IR_TRANSFORMS_UTILS_VAR_VALUE_COLLECTOR_H_
#define SRC_IR_TRANSFORMS_UTILS_VAR_VALUE_COLLECTOR_H_

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::ir {

/// Every value each Var can hold in a function body: what an ``AssignStmt``
/// binds it to (several before SSA), an ``IterArg``'s initial value and each
/// value yielded back into it, and each value a loop or ``if`` result variable
/// receives. A Var absent from the map is bound some other way.
class VarValueCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, std::vector<ExprPtr>> values;

  /// Find the common origin of a selected SSA value on every incoming path.
  /// The classifier returns either a forwarding expression or a terminal origin.
  /// An absent map entry has not been reached yet; Origin{} means unknown or conflicting.
  template <typename Origin, typename Tracks, typename Classify>
  std::unordered_map<const Var*, Origin> CommonOrigins(Tracks tracks, Classify classify) const {
    std::unordered_map<const Var*, Origin> origins;
    std::unordered_map<const Var*, std::vector<const Var*>> users;
    std::vector<const Var*> sources, work;
    auto merge = [&](const Var* var, Origin incoming) {
      auto [it, inserted] = origins.try_emplace(var, incoming);
      if (!inserted) {
        if (it->second == Origin{} || it->second == incoming) return;
        it->second = Origin{};
      }
      work.push_back(var);
    };
    for (const auto& [var, inputs] : values) {
      if (!tracks(var->GetType())) continue;
      for (const auto& input : inputs) {
        auto [source, origin] = classify(input);
        if (source) {
          sources.push_back(source);
          users[source].push_back(var);
        } else {
          merge(var, origin);
        }
      }
    }
    // A value changes at most twice: once on first reaching it, and once if
    // conflicting origins reach it. This also handles loop-carried cycles.
    auto propagate = [&]() {
      while (!work.empty()) {
        auto var = work.back();
        work.pop_back();
        const auto it = users.find(var);
        if (it == users.end()) continue;
        const auto origin = origins.at(var);
        for (const auto* user : it->second) merge(user, origin);
      }
    };
    propagate();
    // Unknown parameters and cycles with no seed must also invalidate downstream joins.
    for (const auto* source : sources) {
      if (!origins.count(source)) merge(source, Origin{});
    }
    propagate();
    return origins;
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) values[op->var_.get()].push_back(op->value_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    yield_targets_.push_back(RecordLoopCarried(op->iter_args_, op->return_vars_));
    IRVisitor::VisitStmt_(op);
    yield_targets_.pop_back();
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    yield_targets_.push_back(RecordLoopCarried(op->iter_args_, op->return_vars_));
    IRVisitor::VisitStmt_(op);
    yield_targets_.pop_back();
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    std::vector<const Var*> targets;
    targets.reserve(op->return_vars_.size());
    for (const auto& return_var : op->return_vars_) targets.push_back(return_var.get());
    yield_targets_.push_back(std::move(targets));
    IRVisitor::VisitStmt_(op);
    yield_targets_.pop_back();
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    if (op && !yield_targets_.empty()) {
      const auto& targets = yield_targets_.back();
      for (size_t i = 0; i < op->value_.size() && i < targets.size(); ++i) {
        if (targets[i]) values[targets[i]].push_back(op->value_[i]);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  /// Per enclosing For / While / If, innermost last: the Vars its yields bind.
  std::vector<std::vector<const Var*>> yield_targets_;

  /// Record a loop's carried values; return the Vars its body's yields bind.
  std::vector<const Var*> RecordLoopCarried(const std::vector<IterArgPtr>& iter_args,
                                            const std::vector<VarPtr>& return_vars) {
    std::vector<const Var*> targets;
    targets.reserve(iter_args.size());
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& iter_arg = iter_args[i];
      targets.push_back(iter_arg.get());
      if (!iter_arg) continue;
      values[iter_arg.get()].push_back(iter_arg->initValue_);
      // The result holds the carried value after the last trip (or the initial
      // value after none), which is exactly what the IterArg can hold.
      if (i < return_vars.size() && return_vars[i]) values[return_vars[i].get()].push_back(iter_arg);
    }
    return targets;
  }
};

}  // namespace pypto::ir

#endif  // SRC_IR_TRANSFORMS_UTILS_VAR_VALUE_COLLECTOR_H_
