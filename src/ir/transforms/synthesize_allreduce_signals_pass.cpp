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
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

[[nodiscard]] bool IsTensorAllReduce(const CallPtr& call) {
  return call && call->op_ && IsOp(call, "pld.tensor.allreduce");
}

[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::HOST) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

// Follow a data Var's SSA def chain back to its alloc-window-buffer LHS Var
// (its "lineage"), so implicit allreduce calls over the same buffer share one
// synthesized signal while distinct buffers (and therefore distinct device
// coverage / comm domains) get distinct signals. Mirrors
// MaterializeCommDomainScopes::ResolveWindowRecord: ``pld.tensor.window``
// resolves through args_[0], push-based collectives (all_to_all / allgather /
// all_to_all_v) resolve through their target arg (args_[1]), and an allreduce
// result aliases its own data arg (args_[0]).
//
// Loop-carried data is an IterArg, not a Var (ConvertToSSA runs first): it has
// no AssignStmt RHS, so follow its init value instead. IterArg has its own
// ObjectKind, so As<Var> misses it — use AsVarLike throughout (see
// .claude/rules/ir-kind-traits.md, same pattern as InferTileMemorySpace #2547).
// Resolution through the init value keys every loop iteration to the carry's
// first-iteration lineage; a loop ping-ponging between two *different-coverage*
// windows is therefore not distinguished (documented limitation).
[[nodiscard]] const Var* ResolveLineageKey(const VarPtr& var,
                                           const std::unordered_map<const Var*, ExprPtr>& var_defs) {
  std::unordered_set<const Var*> visited;
  const Var* cur = var.get();
  while (cur != nullptr && visited.insert(cur).second) {
    if (cur->GetKind() == ObjectKind::IterArg) {
      auto init = static_cast<const IterArg*>(cur)->initValue_;
      if (auto next = AsVarLike(init)) {
        cur = next.get();
        continue;
      }
      return cur;
    }
    auto it = var_defs.find(cur);
    if (it == var_defs.end() || !it->second) return cur;
    const ExprPtr& def = it->second;
    if (auto alias = AsVarLike(def)) {
      // An alias to a loop carry resolves through the carry's init value.
      if (alias->GetKind() == ObjectKind::IterArg) {
        auto init = static_cast<const IterArg*>(alias.get())->initValue_;
        if (auto next = AsVarLike(init)) {
          cur = next.get();
          continue;
        }
        return cur;
      }
      cur = alias.get();
      continue;
    }
    if (auto call = As<Call>(def)) {
      if (call->op_ && IsOp(call, "pld.tensor.window") && !call->args_.empty()) {
        if (auto buf = AsVarLike(call->args_[0])) {
          cur = buf.get();
          continue;
        }
      }
      if (IsTensorAllReduce(call) && !call->args_.empty()) {
        if (auto tgt = AsVarLike(call->args_[0])) {
          cur = tgt.get();
          continue;
        }
      }
      if (call->op_ && (IsOp(call, "pld.tensor.all_to_all") || IsOp(call, "pld.tensor.allgather")) &&
          call->args_.size() > 1) {
        if (auto tgt = AsVarLike(call->args_[1])) {
          cur = tgt.get();
          continue;
        }
      }
      if (call->op_ &&
          (IsOp(call, "pld.tensor.all_to_all_v") || IsOp(call, "pld.tensor.indexed_all_to_all_v")) &&
          call->args_.size() > 1) {
        if (auto tgt = AsVarLike(call->args_[1])) {
          cur = tgt.get();
          continue;
        }
      }
    }
    return cur;
  }
  return cur != nullptr ? cur : var.get();
}

class NameCollector : public IRVisitor {
 public:
  std::set<std::string> names;

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op && !op->name_hint_.empty()) names.insert(op->name_hint_);
    IRVisitor::VisitVarLike_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "pld.tensor.alloc_window_buffer")) {
      auto name = op->GetKwarg<std::string>("name");
      if (!name.empty()) names.insert(name);
    }
    IRVisitor::VisitExpr_(op);
  }
};

// One shared mesh signal per data-buffer lineage group. Each group's binding is
// hoisted to the top of the function body and reused by every implicit-signal
// allreduce call over that buffer (including calls inside for/while loops): the
// host builtin kernels self-clear their barrier cells after each call, so a
// reused signal is correct across back-to-back and loop-carried calls. Distinct
// data buffers get distinct signals, so implicit allreduces over different
// device subsets do not merge into one comm-domain scope.
struct SharedSignalBinding {
  std::vector<StmtPtr> prefix;  ///< world_size / alloc_window_buffer / window assigns
  VarPtr signal_var;
};

struct SignalNames {
  std::string world_size_name;
  std::string buf_name;
  std::string signal_name;
};

[[nodiscard]] SignalNames FreshSignalNames(std::set<std::string>* used_names, int64_t* next_id) {
  while (true) {
    auto suffix = std::to_string((*next_id)++);
    SignalNames names{"__allreduce_signal_world_size_" + suffix, "__allreduce_signal_buf_" + suffix,
                      "__allreduce_signal_" + suffix};
    if (used_names->count(names.world_size_name) != 0 || used_names->count(names.buf_name) != 0 ||
        used_names->count(names.signal_name) != 0) {
      continue;
    }
    used_names->insert(names.world_size_name);
    used_names->insert(names.buf_name);
    used_names->insert(names.signal_name);
    return names;
  }
}

[[nodiscard]] SharedSignalBinding MakeSharedSignalBinding(std::set<std::string>* used_names, int64_t* next_id,
                                                          const Span& span, int core_num) {
  auto names = FreshSignalNames(used_names, next_id);
  INTERNAL_CHECK_SPAN(core_num > 0, span)
      << "SynthesizeAllReduceSignals requires a positive allreduce core_num";

  auto world_size_call = OpRegistry::GetInstance().Create("pld.system.world_size", {}, span);
  auto world_size_var = std::make_shared<Var>(names.world_size_name, world_size_call->GetType(), span);
  auto world_size_assign = std::make_shared<AssignStmt>(world_size_var, world_size_call, span);

  auto core_num_expr = std::make_shared<ConstInt>(core_num, DataType::INT64, span);
  auto four = std::make_shared<ConstInt>(4, DataType::INT64, span);
  auto signal_elements = MakeMul(world_size_var, core_num_expr, span);
  auto size_bytes = MakeMul(signal_elements, four, span);

  std::vector<std::pair<std::string, std::any>> alloc_kwargs = {{"name", names.buf_name}};
  auto alloc_call =
      OpRegistry::GetInstance().Create("pld.tensor.alloc_window_buffer", {size_bytes}, alloc_kwargs, span);
  auto buf_var = std::make_shared<Var>(names.buf_name, alloc_call->GetType(), span);
  auto buf_assign = std::make_shared<AssignStmt>(buf_var, alloc_call, span);

  auto signal_shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{world_size_var, core_num_expr}, span);
  std::vector<std::pair<std::string, std::any>> window_kwargs = {{"dtype", DataType::INT32}};
  auto window_call =
      OpRegistry::GetInstance().Create("pld.tensor.window", {buf_var, signal_shape}, window_kwargs, span);
  auto signal_var = std::make_shared<Var>(names.signal_name, window_call->GetType(), span);
  auto signal_assign = std::make_shared<AssignStmt>(signal_var, window_call, span);

  return {{world_size_assign, buf_assign, signal_assign}, signal_var};
}

/// Pre-scan: does this host_orch function need the synthesizer at all?
///
/// The synthesizer runs on any function carrying a ``pld.tensor.allreduce``
/// (it also lifts return-position calls and rejects nested calls for
/// explicit-signal functions); only a 1-arg call additionally needs a shared
/// signal binding.
class AllReduceSignalNeedFinder : public IRVisitor {
 public:
  bool has_allreduce = false;                           ///< any pld.tensor.allreduce call
  std::vector<std::pair<ExprPtr, int>> implicit_calls;  ///< 1-arg calls in visit order: {data, core_num}
  std::unordered_map<const Var*, ExprPtr> var_defs;     ///< Var* -> defining RHS (lineage resolution)

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (IsTensorAllReduce(op)) {
      has_allreduce = true;
      if (op->args_.size() == 1) {
        implicit_calls.emplace_back(op->args_[0], op->GetKwarg<int>("core_num"));
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // AsVarLike (not As<Var>): an SSA loop-carry LHS is an IterArg and must be
    // resolvable back through ResolveLineageKey (ir-kind-traits.md).
    if (auto var = AsVarLike(op->var_)) {
      var_defs[var.get()] = op->value_;
    }
    IRVisitor::VisitStmt_(op);
  }
};

class AllReduceSignalSynthesizer : public IRMutator {
 public:
  using SignalLookup = std::function<VarPtr(const ExprPtr&)>;

  AllReduceSignalSynthesizer(std::set<std::string>* used_names, int64_t* next_id, SignalLookup signal_lookup)
      : used_names_(used_names), next_id_(next_id), signal_lookup_(std::move(signal_lookup)) {}

  [[nodiscard]] bool modified() const { return modified_; }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    if (IsTensorAllReduce(op)) {
      CheckAllReduceCall(op);
      CHECK_SPAN(false, op->span_)
          << "pld.tensor.allreduce must be a direct assignment, expression statement, or return value before "
             "allreduce signal synthesis.";
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!IsTensorAllReduce(call)) return IRMutator::VisitStmt_(op);
    CheckAllReduceCall(call);
    if (call->args_.size() == 2) {
      for (const auto& arg : call->args_) VisitExpr(arg);
      return op;
    }

    auto target = VisitExpr(call->args_[0]);
    auto rewritten_call = MakeAllReduceCall(call, target, signal_lookup_(call->args_[0]));
    auto result = MutableCopy(op);
    result->value_ = rewritten_call;
    modified_ = true;
    return result;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!IsTensorAllReduce(call)) return IRMutator::VisitStmt_(op);
    CheckAllReduceCall(call);
    if (call->args_.size() == 2) {
      for (const auto& arg : call->args_) VisitExpr(arg);
      return op;
    }

    auto target = VisitExpr(call->args_[0]);
    auto rewritten_call = MakeAllReduceCall(call, target, signal_lookup_(call->args_[0]));
    modified_ = true;
    return std::make_shared<EvalStmt>(rewritten_call, op->span_, op->leading_comments_);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<StmtPtr> prelude;
    std::vector<ExprPtr> new_values;
    new_values.reserve(op->value_.size());
    bool changed = false;

    for (std::size_t i = 0; i < op->value_.size(); ++i) {
      INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
      auto call = As<Call>(op->value_[i]);
      if (!IsTensorAllReduce(call)) {
        auto new_value = VisitExpr(op->value_[i]);
        new_values.push_back(new_value);
        if (new_value.get() != op->value_[i].get()) changed = true;
        continue;
      }

      CheckAllReduceCall(call);
      auto target = VisitExpr(call->args_[0]);
      auto signal = call->args_.size() == 1 ? signal_lookup_(call->args_[0]) : VisitExpr(call->args_[1]);
      auto rewritten_call = MakeAllReduceCall(call, target, signal);
      auto result_var = std::make_shared<Var>(FreshGeneratedName("__allreduce_result_"),
                                              rewritten_call->GetType(), call->span_);
      prelude.push_back(std::make_shared<AssignStmt>(result_var, rewritten_call, call->span_));
      new_values.push_back(result_var);
      changed = true;
    }

    if (!changed) return op;
    auto new_return = MutableCopy(op);
    new_return->value_ = std::move(new_values);
    prelude.push_back(new_return);
    modified_ = true;
    return SeqStmts::Flatten(std::move(prelude), op->span_);
  }

 private:
  void CheckAllReduceCall(const CallPtr& call) const {
    CHECK_SPAN(call->args_.size() == 1 || call->args_.size() == 2, call->span_)
        << "pld.tensor.allreduce expects target[, signal], got " << call->args_.size()
        << " positional arguments";
  }

  [[nodiscard]] std::string FreshGeneratedName(const std::string& prefix) {
    while (true) {
      auto name = prefix + std::to_string((*next_id_)++);
      if (used_names_->count(name) != 0) continue;
      used_names_->insert(name);
      return name;
    }
  }

  [[nodiscard]] CallPtr MakeAllReduceCall(const CallPtr& call, const ExprPtr& target, const ExprPtr& signal) {
    return OpRegistry::GetInstance().Create("pld.tensor.allreduce", {target, signal}, call->kwargs_,
                                            call->span_);
  }

  std::set<std::string>* used_names_;
  int64_t* next_id_;
  SignalLookup signal_lookup_;
  bool modified_ = false;
};

}  // namespace

namespace pass {

Pass SynthesizeAllReduceSignals() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    NameCollector name_collector;
    name_collector.VisitProgram(program);
    int64_t next_signal_id = 0;

    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool modified = false;
    for (const auto& [gvar, func] : program->functions_) {
      if (!IsHostOrch(func)) {
        new_functions[gvar] = func;
        continue;
      }

      // One shared mesh signal per data-buffer lineage group, hoisted before the
      // body and reused by every implicit-signal call over that buffer (incl.
      // inside loops — self-clearing kernels make reuse safe). Distinct data
      // buffers get distinct signals, so implicit allreduces over different
      // device subsets do not merge into a single comm-domain scope. The
      // synthesizer still runs for explicit-signal functions to lift
      // return-position calls and reject nested calls.
      AllReduceSignalNeedFinder finder;
      finder.VisitStmt(func->body_);
      if (!finder.has_allreduce) {
        new_functions[gvar] = func;
        continue;
      }

      std::vector<const void*> lineage_order;
      std::unordered_map<const void*, int> lineage_core_num;
      std::unordered_map<const void*, VarPtr> lineage_to_signal;
      std::vector<StmtPtr> body_stmts;
      for (const auto& [data_expr, core_num] : finder.implicit_calls) {
        const void* key;
        // AsVarLike (not As<Var>): loop-carried data is an IterArg and must
        // resolve through its lineage group, not fall back to a per-call
        // signal (ir-kind-traits.md).
        if (auto data_var = AsVarLike(data_expr)) {
          key = static_cast<const void*>(ResolveLineageKey(data_var, finder.var_defs));
        } else {
          key = static_cast<const void*>(data_expr.get());  // non-Var data → per-call signal
        }
        auto [it, inserted] = lineage_core_num.emplace(key, core_num);
        if (!inserted) {
          it->second = std::max(it->second, core_num);
        } else {
          lineage_order.push_back(key);
        }
      }
      for (const void* key : lineage_order) {
        auto binding = MakeSharedSignalBinding(&name_collector.names, &next_signal_id, func->span_,
                                               lineage_core_num[key]);
        lineage_to_signal[key] = binding.signal_var;
        body_stmts.insert(body_stmts.end(), binding.prefix.begin(), binding.prefix.end());
      }
      StmtPtr body_to_visit;
      if (body_stmts.empty()) {
        body_to_visit = func->body_;
      } else {
        body_stmts.push_back(func->body_);
        body_to_visit = SeqStmts::Flatten(std::move(body_stmts), func->span_);
      }

      AllReduceSignalSynthesizer::SignalLookup signal_lookup = [&lineage_to_signal,
                                                                &finder](const ExprPtr& data) -> VarPtr {
        const void* key;
        // AsVarLike (not As<Var>): loop-carried data is an IterArg and must
        // resolve through its lineage group, not fall back to a per-call
        // signal (ir-kind-traits.md).
        if (auto data_var = AsVarLike(data)) {
          key = static_cast<const void*>(ResolveLineageKey(data_var, finder.var_defs));
        } else {
          key = static_cast<const void*>(data.get());
        }
        auto it = lineage_to_signal.find(key);
        INTERNAL_CHECK(it != lineage_to_signal.end())
            << "SynthesizeAllReduceSignals: no synthesized signal for allreduce data lineage";
        return it->second;
      };

      AllReduceSignalSynthesizer synthesizer(&name_collector.names, &next_signal_id, signal_lookup);
      auto new_body = synthesizer.VisitStmt(body_to_visit);
      if (!synthesizer.modified()) {
        new_functions[gvar] = func;
        continue;
      }

      auto new_func = MutableCopy(func);
      new_func->body_ = new_body;
      new_functions[gvar] = new_func;
      modified = true;
    }

    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "SynthesizeAllReduceSignals", kSynthesizeAllReduceSignalsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
