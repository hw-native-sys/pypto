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
#include <array>
#include <cstddef>
#include <cstdint>
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

#include "pypto/core/error.h"
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
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/collective_operand_utils.h"
#include "pypto/ir/transforms/utils/dist_tensor_alias_utils.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/window_view_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

[[nodiscard]] bool IsTensorAllReduce(const CallPtr& call) {
  return call && call->op_ && IsOp(call, "pld.tensor.allreduce");
}

// pld.tensor.all_to_all(input, target, signal) and
// pld.tensor.allgather(input, target, signal) return `target` in-place
// (args_[1]) — unlike allreduce, which aliases args_[0]. A window view
// bound to such a call's result must resolve back through this arg
// index so the target allocation's device coverage is inferred correctly
// even when no dispatch site references the pre-call view directly (e.g.
// when input and target are two distinct buffers, as they must be — see
// kernel.cpp.in for why aliasing them is unsafe).
[[nodiscard]] bool IsTensorAllToAll(const CallPtr& call) {
  return call && call->op_ && IsOp(call, "pld.tensor.all_to_all");
}

[[nodiscard]] bool IsTensorAllGather(const CallPtr& call) {
  return call && call->op_ && IsOp(call, "pld.tensor.allgather");
}

// pld.tensor.all_to_all_v(input, target, signal, send_counts, recv_counts)
// also returns `target` in-place (args_[1]) — same aliasing discipline as
// IsTensorAllToAll above.
[[nodiscard]] bool IsTensorAllToAllV(const CallPtr& call) {
  return call && call->op_ &&
         (IsOp(call, "pld.tensor.all_to_all_v") || IsOp(call, "pld.tensor.indexed_all_to_all_v"));
}

/// Device coverage descriptor inferred from a dispatch ``device=`` expression.
struct DeviceDescriptor {
  bool is_all = false;
  std::set<int64_t> subset;

  bool operator==(const DeviceDescriptor& o) const {
    return is_all == o.is_all && (is_all || subset == o.subset);
  }
  bool operator<(const DeviceDescriptor& o) const {
    if (is_all != o.is_all) return is_all < o.is_all;
    if (is_all) return false;  // all-device: subset is meaningless, matches operator==
    return subset < o.subset;
  }

  void Merge(const DeviceDescriptor& other) {
    if (is_all || other.is_all) {
      is_all = true;
      subset.clear();
      return;
    }
    subset.insert(other.subset.begin(), other.subset.end());
  }

  [[nodiscard]] std::vector<int64_t> ToDevices() const {
    if (is_all) return {};
    return {subset.begin(), subset.end()};
  }
};

/// Per-alloc bookkeeping populated during the host_orch scan.
struct AllocRecord {
  CallPtr alloc_call;                  ///< pld.tensor.alloc_window_buffer Call
  VarPtr ptr_var;                      ///< AssignStmt LHS (Var of PtrType)
  ExprPtr size_expr;                   ///< alloc_call->args_[0]
  std::string name;                    ///< from alloc_call attr "name"
  Span span;                           ///< alloc_call->span_ (const fields → emplaced)
  std::vector<DeviceDescriptor> seen;  ///< one per consuming dispatch
  WindowBufferPtr wb;                  ///< filled after construction

  AllocRecord(CallPtr ac, VarPtr pv, ExprPtr sz, std::string nm, Span sp)
      : alloc_call(std::move(ac)),
        ptr_var(std::move(pv)),
        size_expr(std::move(sz)),
        name(std::move(nm)),
        span(std::move(sp)) {}
};

/// Per-pld.tensor.window result Var: maps the LHS Var pointer back to its alloc.
struct WindowRecord {
  CallPtr window_call;
  VarPtr old_view_var;
  AllocRecord* alloc;
};

struct CollectiveConsumer {
  AllocRecord* data_alloc;  ///< nullptr for barrier-only consumers
  AllocRecord* signal_alloc;
  Span span;
};

/// Scans a host_orch function body once and records every
/// ``pld.tensor.alloc_window_buffer`` and ``pld.tensor.window`` assignment.
class AllocAndWindowCollector : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    auto var = AsVarLike(op->var_);
    auto call = As<Call>(op->value_);
    if (var && call && call->op_) {
      if (IsOp(call, "pld.tensor.alloc_window_buffer")) {
        INTERNAL_CHECK_SPAN(call->args_.size() == 1, call->span_)
            << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer expects exactly one arg (size)";
        // The parser injects ``name`` as a kwarg derived from the assignment
        // LHS — not as an ``attrs`` entry — so use GetKwarg here.
        auto name = call->GetKwarg<std::string>("name");
        INTERNAL_CHECK_SPAN(!name.empty(), call->span_)
            << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer missing 'name' kwarg";
        auto rec = std::make_unique<AllocRecord>(call, var, call->args_[0], name, call->span_);
        ptr_to_alloc[var.get()] = rec.get();
        allocs.push_back(std::move(rec));
      } else if (IsOp(call, "pld.tensor.window") && !call->args_.empty()) {
        // AsVarLike (not As<Var>): the windowed buffer may be an SSA loop carry
        // (IterArg) — record it so ResolveWindowRecord can follow it.
        auto ptr_arg_var = AsVarLike(call->args_[0]);
        if (ptr_arg_var) {
          auto it = ptr_to_alloc.find(ptr_arg_var.get());
          if (it != ptr_to_alloc.end()) {
            WindowRecord wr{call, var, it->second};
            view_to_window[var.get()] = wr;
            windows.push_back(wr);
          }
        }
      } else if (IsTensorAllReduce(call) && !call->args_.empty()) {
        // An allreduce result aliases its data window: record (result, data)
        // so the substitution below can re-type the result with the same
        // WindowBuffer (loop-carried results must keep window identity).
        collective_results.emplace_back(var, AsVarLike(call->args_[0]));
      }
    }
    // Record the AssignStmt def for every Var so ResolveDeviceDescriptor can
    // follow ``for r in pl.range(<var>)`` back to ``<var> = pld.system.world_size()``
    // (CSE / NormalizeStmtStructure hoists such calls out into a temp).
    // AsVarLike (not As<Var>): an SSA loop-carry LHS is an IterArg.
    if (auto var = AsVarLike(op->var_)) {
      var_defs[var.get()] = op->value_;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op->iter_args_.empty()) loops.push_back(op);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op->iter_args_.empty()) whiles.push_back(op);
    IRVisitor::VisitStmt_(op);
  }

  std::vector<std::unique_ptr<AllocRecord>> allocs;
  std::unordered_map<const Var*, AllocRecord*> ptr_to_alloc;
  std::unordered_map<const Var*, WindowRecord> view_to_window;
  std::vector<WindowRecord> windows;
  std::vector<std::pair<VarPtr, VarPtr>> collective_results;  ///< {allreduce result, data arg}
  std::vector<ForStmtPtr> loops;                              ///< loops carrying values (iter_args non-empty)
  std::vector<WhileStmtPtr> whiles;
  std::unordered_map<const Var*, ExprPtr> var_defs;
};

// Index 0 is URMA; index 1 is MTE. MTE is an analysis-only requirement:
// its runtime domain uses an empty workspace-engine list.
using BufferEngines = std::unordered_map<const Var*, std::array<bool, 2>>;

std::array<bool, 2> EnginesFor(const BufferEngines& engines, const Var* buffer) {
  const auto it = engines.find(buffer);
  return it == engines.end() ? std::array<bool, 2>{} : it->second;
}

// Collect effects per allocation, before compatible allocations are grouped.
// Propagate callee requirements back to each actual argument independently;
// caller facts never flow into shared formals and contaminate another call.
// Each engine bit traverses an edge at most once: O(nodes + edges).
class BufferEngineCollector : public IRVisitor {
 public:
  explicit BufferEngineCollector(ProgramPtr program) : program_(std::move(program)) {}

  BufferEngines Collect() {
    // Standalone invocations need not have normalized return statements yet.
    // Share the existing memoized lineage analysis across all callee queries.
    return_lineage::ScopedTraceCache lineage_cache;
    for (const auto& [gv, func] : program_->functions_) {
      if (func->func_type_ == FunctionType::SIMT) continue;
      auto& plan = plans_[func->name_];
      plan.returned_param_indices = return_lineage::ReturnedParamIndices(func, program_);
      for (size_t i = 0; i < func->params_.size(); ++i) {
        const auto& param = func->params_[i];
        if (!As<DistributedTensorType>(param->GetType())) continue;
        plan.dist_param_indices.push_back(i);
        plan.param_to_ctx[param.get()] = param;
      }
    }
    for (const auto& [gv, func] : program_->functions_) {
      auto plan = plans_.find(func->name_);
      dist_tensor_alias::DistParamAliasCollector aliases(program_, &plans_,
                                                         plan == plans_.end() ? nullptr : &plan->second);
      AllocAndWindowCollector windows;
      windows.VisitStmt(func->body_);
      for (const auto& alloc : windows.allocs) roots_[alloc->ptr_var.get()] = alloc->ptr_var;
      // Seed raw views with their allocation identity. The same alias walker
      // used by context materialization handles returns, loop carries and views.
      for (const auto& view : windows.windows) {
        aliases.alias_to_ctx[view.old_view_var.get()] = view.alloc->ptr_var;
      }
      aliases.VisitStmt(func->body_);
      aliases_ = &aliases;
      VisitStmt(func->body_);
    }
    aliases_ = nullptr;
    // Record may append work; consume by index without retaining iterators.
    size_t next = 0;
    while (next < pending_.size()) {
      const auto [node, engine] = pending_[next++];
      for (const auto* caller : callers_[node]) Record(caller, engine);
    }
    if (roots_.empty()) {
      // A standalone CHIP entry has no allocation sites yet, but incompatible
      // uses of the same formal still necessarily refer to one buffer.
      for (const auto& [buffer, engines] : engines_) Check(buffer, engines);
    } else {
      for (const auto& [buffer, owner] : roots_) Check(buffer, EnginesFor(engines_, buffer));
    }
    return std::move(engines_);
  }

 protected:
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    // Re-running the pass must retain allocation identity even for explicit
    // materialized IR whose body contains no alloc_window_buffer calls.
    for (const auto& slot : op->slots_) roots_[slot->base_.get()] = slot->base_;
    IRVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallPtr& op) override {
    Analyze(op->op_, op->args_, op->span_);
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    Analyze(op->op_, op->args_, op->span_);
    IRVisitor::VisitExpr_(op);
  }

 private:
  const Var* Resolve(const ExprPtr& expr) const {
    if (auto type = As<DistributedTensorType>(expr->GetType()); type && type->window_buffer_) {
      return (*type->window_buffer_)->base_.get();
    }
    return aliases_->ResolveCtx(expr).get();
  }

  void Analyze(const OpPtr& op, const std::vector<ExprPtr>& args, const Span& span) {
    if (auto roles = GetCollectiveOperandRoles(op)) {
      for (size_t index : roles->context_args) {
        INTERNAL_CHECK_SPAN(index < args.size(), span) << "Collective is missing a remote operand";
        if (const auto* buffer = Resolve(args[index])) {
          Record(buffer, roles->target_uses_urma && index == roles->context_args.front() ? 0 : 1);
        }
      }
    }
    if (auto global = As<GlobalVar>(op)) {
      auto callee = program_->GetFunction(global->name_);
      auto plan = plans_.find(global->name_);
      if (!callee || plan == plans_.end()) return;
      // Before context suffixes are appended, Submit supplies an ordinary
      // parameter prefix; omitted runtime-owned outputs have no actual buffer.
      for (size_t index : plan->second.dist_param_indices) {
        if (index >= args.size()) continue;
        if (const auto* actual = Resolve(args[index])) {
          callers_[callee->params_[index].get()].push_back(actual);
        }
      }
      return;
    }
    size_t target = 0;
    const bool urma = IsOp(op, "pld.tensor.put_async") || IsOp(op, "pld.tensor.get_async");
    if (IsOp(op, "pld.tensor.get_async") || IsOp(op, "pld.tensor.get") || IsOp(op, "pld.tile.get")) {
      target = 2;
    } else if (IsOp(op, "pld.tensor.remote_store") || IsOp(op, "pld.tile.remote_store")) {
      target = 1;
    } else if (!urma && !IsOp(op, "pld.system.notify") && !IsOp(op, "pld.tile.remote_load") &&
               !IsOp(op, "pld.tensor.put") && !IsOp(op, "pld.tile.put")) {
      return;
    }
    INTERNAL_CHECK_SPAN(args.size() > target, span)
        << "Communication primitive is missing its window operand";
    if (const auto* buffer = Resolve(args[target])) Record(buffer, urma ? 0 : 1);
  }

  void Record(const Var* buffer, size_t engine) {
    if (engines_[buffer][engine]) return;
    engines_[buffer][engine] = true;
    pending_.emplace_back(buffer, engine);
  }
  static void Check(const Var* buffer, const std::array<bool, 2>& engines) {
    CHECK_SPAN(!engines[0] || !engines[1], buffer->span_)
        << "Communication buffer '" << buffer->name_hint_
        << "' cannot combine URMA async transfers with MTE primitives "
           "(remote_load/store, put/get, notify or collective operands), regardless of peer rank.";
  }

  ProgramPtr program_;
  std::unordered_map<std::string, dist_tensor_alias::FunctionCtxPlan> plans_;
  const dist_tensor_alias::DistParamAliasCollector* aliases_ = nullptr;
  std::unordered_map<const Var*, VarPtr> roots_;
  std::unordered_map<const Var*, std::vector<const Var*>> callers_;
  BufferEngines engines_;
  std::vector<std::pair<const Var*, size_t>> pending_;
};

/// Detects whether a Call resolves to a chip-level Orchestration function (i.e.
/// the host_orch is dispatching down one level). Such calls carry the
/// ``device=`` attr written by the N3 parser pass.
[[nodiscard]] bool IsChipOrchDispatch(const CallPtr& op,
                                      const std::map<std::string, FunctionPtr>& chip_orchs) {
  if (!op || !op->op_) return false;
  auto gvar = As<GlobalVar>(op->op_);
  if (!gvar) return false;
  return chip_orchs.find(gvar->name_) != chip_orchs.end();
}

/// Unwrap a ``stop_`` expression through one level of SSA assignment indirection
/// so the dispatch device resolver can see through CSE-hoisted bounds like
/// ``t__tmp_v0 = pld.system.world_size(); for r in pl.range(t__tmp_v0):``.
/// Returns ``stop`` unchanged when it is already a literal/call or when the
/// chain dead-ends in a Var without a known def.
[[nodiscard]] ExprPtr UnwrapStopExpr(const ExprPtr& stop,
                                     const std::unordered_map<const Var*, ExprPtr>& var_defs) {
  ExprPtr cur = stop;
  std::unordered_set<const Var*> visited;
  while (auto v = As<Var>(cur)) {
    if (!visited.insert(v.get()).second) return cur;
    auto it = var_defs.find(v.get());
    if (it == var_defs.end() || !it->second) return cur;
    cur = it->second;
  }
  return cur;
}

/// Resolves the device descriptor for a ``device=`` Expr in the context of a
/// stack of enclosing ForStmt scopes. Throws pypto::ValueError on unsupported
/// forms (the user's parser is meant to restrict ``device=`` to ConstInt or
/// the induction var of an enclosing pl.range loop).
DeviceDescriptor ResolveDeviceDescriptor(const ExprPtr& device, const std::vector<ForStmtPtr>& for_stack,
                                         const std::unordered_map<const Var*, ExprPtr>& var_defs,
                                         const Span& dispatch_span) {
  DeviceDescriptor desc;
  if (auto ci = As<ConstInt>(device)) {
    INTERNAL_CHECK_SPAN(ci->value_ >= 0, dispatch_span)
        << "MaterializeCommDomainScopes: device= ConstInt must be non-negative, got " << ci->value_;
    desc.subset.insert(ci->value_);
    return desc;
  }
  if (auto v = As<Var>(device)) {
    for (auto it = for_stack.rbegin(); it != for_stack.rend(); ++it) {
      const auto& fs = *it;
      if (fs->loop_var_.get() == v.get()) {
        // Loop bound determines coverage. Unwrap one level of SSA-assigned temp
        // so a hoisted ``t = pld.system.world_size()`` is recognised the same
        // as the direct ``pl.range(pld.system.world_size())`` form.
        ExprPtr stop = UnwrapStopExpr(fs->stop_, var_defs);
        if (auto stop_call = As<Call>(stop)) {
          if (stop_call->op_ && IsOp(stop_call, "pld.system.world_size")) {
            desc.is_all = true;
            return desc;
          }
        }
        if (auto stop_ci = As<ConstInt>(stop)) {
          auto start_ci = As<ConstInt>(UnwrapStopExpr(fs->start_, var_defs));
          INTERNAL_CHECK_SPAN(start_ci, dispatch_span)
              << "MaterializeCommDomainScopes: device=r loop start must unwrap to ConstInt";
          int64_t start = start_ci->value_;
          auto step_ci = As<ConstInt>(UnwrapStopExpr(fs->step_, var_defs));
          INTERNAL_CHECK_SPAN(step_ci, dispatch_span)
              << "MaterializeCommDomainScopes: device=r loop step must unwrap to ConstInt";
          int64_t step = step_ci->value_;
          INTERNAL_CHECK_SPAN(step == 1, dispatch_span)
              << "MaterializeCommDomainScopes: device=r over a non-unit-step loop is not supported (step="
              << step << ")";
          INTERNAL_CHECK_SPAN(start >= 0 && stop_ci->value_ >= start, dispatch_span)
              << "MaterializeCommDomainScopes: device=r loop range must be [0, N) with N>=0";
          for (int64_t i = start; i < stop_ci->value_; ++i) desc.subset.insert(i);
          return desc;
        }
        throw pypto::ValueError(
            "MaterializeCommDomainScopes: device=r loop bound must be ConstInt or pld.system.world_size()");
      }
    }
    throw pypto::ValueError(
        "MaterializeCommDomainScopes: device= Var is not the induction variable of any enclosing pl.range "
        "loop");
  }
  throw pypto::ValueError(
      "MaterializeCommDomainScopes: device= expression must be ConstInt or the induction var of pl.range; "
      "got "
      "an unsupported expression at " +
      dispatch_span.to_string());
}

/// Walks a host_orch body, maintaining a stack of enclosing ForStmts, and for
/// every chip_orch dispatch Call records the inferred device descriptor against
/// each view Var passed positionally.
class DispatchAnalyzer : public IRVisitor {
 public:
  DispatchAnalyzer(const std::unordered_map<const Var*, WindowRecord>& view_to_window,
                   const std::map<std::string, FunctionPtr>& chip_orchs,
                   const std::unordered_map<const Var*, ExprPtr>& var_defs)
      : view_to_window_(view_to_window), chip_orchs_(chip_orchs), var_defs_(var_defs) {}

  void VisitStmt_(const ForStmtPtr& op) override {
    for_stack_.push_back(op);
    ++repeating_scope_depth_;
    IRVisitor::VisitStmt_(op);
    --repeating_scope_depth_;
    for_stack_.pop_back();
  }

  // WhileStmt carries no device-descriptor role (unlike ForStmt, which
  // for_stack_ tracks for ResolveDeviceDescriptor), but it must still count
  // toward repeating_scope_depth_ so the all_to_all_v loop-use guard in
  // AnalyzeCollective rejects a call nested in a while loop too.
  void VisitStmt_(const WhileStmtPtr& op) override {
    ++repeating_scope_depth_;
    IRVisitor::VisitStmt_(op);
    --repeating_scope_depth_;
  }

  void AnalyzeDispatch(const CallPtr& op) {
    if (!IsChipOrchDispatch(op, chip_orchs_)) return;
    ExprPtr device;
    for (const auto& [k, v] : op->attrs_) {
      if (k == kAttrDevice) {
        // attrs["device"] is stored as ExprPtr by N3 parser.
        if (const auto* p = std::any_cast<ExprPtr>(&v)) device = *p;
        break;
      }
    }
    if (!device) return;
    DeviceDescriptor desc = ResolveDeviceDescriptor(device, for_stack_, var_defs_, op->span_);
    for (const auto& arg : op->args_) {
      auto arg_var = AsVarLike(arg);
      if (!arg_var) continue;
      auto window = ResolveWindowRecord(arg_var);
      if (window) {
        window->alloc->seen.push_back(desc);
      }
    }
  }

  [[nodiscard]] AllocRecord* ResolveWindowAlloc(const ExprPtr& expr, const std::string& op_name,
                                                const char* role) {
    // AsVarLike (not As<Var>): a loop-carried data/signal arg is an IterArg
    // (ConvertToSSA runs first); ResolveWindowRecord follows its init value.
    auto view_var = AsVarLike(expr);
    INTERNAL_CHECK_SPAN(view_var, expr->span_)
        << "MaterializeCommDomainScopes: " << op_name << " " << role << " must be a window view Var";
    auto window = ResolveWindowRecord(view_var);
    INTERNAL_CHECK_SPAN(window, expr->span_) << "MaterializeCommDomainScopes: " << op_name << " " << role
                                             << " must be produced by pld.tensor.window";
    return window->alloc;
  }

  // Nullable variant of ResolveWindowAlloc for operands that are window-bound
  // on the builtin rails but may legally stay plain Tensors on the InCore
  // composite rail (send_counts today): returns nullptr instead of failing
  // when the expression is not a pld.tensor.window view.
  [[nodiscard]] AllocRecord* TryResolveWindowAlloc(const ExprPtr& expr) {
    auto view_var = AsVarLike(expr);
    if (!view_var) return nullptr;
    auto window = ResolveWindowRecord(view_var);
    return window ? window->alloc : nullptr;
  }

  void AnalyzeCollective(const CallPtr& op) {
    if (!op || !op->op_) return;
    CHECK_SPAN(!IsOp(op, "pld.tensor.indexed_all_to_all_v"), op->span_)
        << "indexed_all_to_all_v supports only L2 orchestration on A5 URMA; HOST is unsupported";

    if (IsOp(op, "pld.tensor.allreduce") || IsOp(op, "pld.tensor.broadcast") ||
        IsOp(op, "pld.tensor.reduce_scatter")) {
      const auto& op_name = op->op_->name_;
      INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_)
          << "MaterializeCommDomainScopes: " << op_name << " expects exactly two args";
      collective_consumers.push_back({ResolveWindowAlloc(op->args_[0], op_name, "data/target"),
                                      ResolveWindowAlloc(op->args_[1], op_name, "signal"), op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.barrier")) {
      INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.barrier expects exactly one arg";
      collective_consumers.push_back(
          {nullptr, ResolveWindowAlloc(op->args_[0], "pld.tensor.barrier", "signal"), op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.allgather")) {
      // 3-arg push-based form: pld.tensor.allgather(input, target, signal).
      // Track target + signal for device-coverage inheritance; input coverage
      // comes from the caller's own dispatch site.
      INTERNAL_CHECK_SPAN(op->args_.size() == 3, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.allgather expects exactly 3 args";
      collective_consumers.push_back({ResolveWindowAlloc(op->args_[1], "pld.tensor.allgather", "target"),
                                      ResolveWindowAlloc(op->args_[2], "pld.tensor.allgather", "signal"),
                                      op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.all_to_all")) {
      // 3-arg push-based form: pld.tensor.all_to_all(input, target, signal)
      // args[0] = input  (Tensor or DistributedTensor — on HOST path this
      //                    must be a DISTINCT window from args[1], never the
      //                    same one; on InCore it is a plain Tensor.)
      // args[1] = target (DistributedTensor, window-bound)
      // args[2] = signal (DistributedTensor, window-bound)
      INTERNAL_CHECK_SPAN(op->args_.size() == 3, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.all_to_all expects exactly 3 args";
      auto* data_alloc = ResolveWindowAlloc(op->args_[1], "pld.tensor.all_to_all", "target");
      auto* signal_alloc = ResolveWindowAlloc(op->args_[2], "pld.tensor.all_to_all", "signal");
      // Device-coverage inheritance from data to signal is handled generically
      // in the Phase 3 loop below (every collective consumer with a data_alloc
      // triggers the merge automatically).  No manual merge needed here.
      collective_consumers.push_back({data_alloc, signal_alloc, op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.all_to_all_v")) {
      // 6-arg push-based form:
      // pld.tensor.all_to_all_v(input, target, signal, send_counts, recv_counts, core_num)
      // args[0] = input        (window-bound on HOST path; distinct from target)
      // args[1] = target       (DistributedTensor, window-bound, window-as-result)
      // args[2] = signal       (DistributedTensor, window-bound, barrier)
      // args[3] = send_counts  (window-bound on the builtin rails; peers READ
      //                          it remotely via the counts pull, so — like
      //                          `signal` and `recv_counts` — it inherits
      //                          target's device coverage; conditional,
      //                          because the InCore composite rail may pass a
      //                          plain Tensor, written locally only)
      // args[4] = recv_counts  (DistributedTensor, window-bound; published
      //                          cross-rank via notify, so needs target's
      //                          device coverage exactly like `signal` does)
      // args[5] = core_num     (Scalar[INDEX], not window-bound — irrelevant
      //                          to this pass's comm-domain scoping)
      INTERNAL_CHECK_SPAN(op->args_.size() == 6, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.all_to_all_v expects exactly 6 args";
      // HOST-orch calls still need loop-carried comm-domain window lifetime
      // management, even though the credit barrier supports rank-uniform
      // InCore loops. Track HOST repetition across both ForStmt and WhileStmt
      // with repeating_scope_depth_ (see the VisitStmt_ overrides above).
      CHECK_SPAN(repeating_scope_depth_ == 0, op->span_)
          << "pld.tensor.all_to_all_v is not supported inside a for/while loop in a HOST "
             "orchestrator yet: dynamic re-invocation needs loop-carried comm-domain window "
             "lifetime management this compiler does not model (the credit-based signal "
             "itself is reusable across sequential calls and rank-uniform InCore loops).";
      auto* data_alloc = ResolveWindowAlloc(op->args_[1], "pld.tensor.all_to_all_v", "target");
      auto* signal_alloc = ResolveWindowAlloc(op->args_[2], "pld.tensor.all_to_all_v", "signal");
      auto* recv_counts_alloc = ResolveWindowAlloc(op->args_[4], "pld.tensor.all_to_all_v", "recv_counts");
      // Consumer entries sharing the same data_alloc: signal, recv_counts and
      // (when window-bound) send_counts inherit target's device coverage
      // (Phase 3 loop below merges generically per entry).
      collective_consumers.push_back({data_alloc, signal_alloc, op->span_});
      collective_consumers.push_back({data_alloc, recv_counts_alloc, op->span_});
      if (auto* send_counts_alloc = TryResolveWindowAlloc(op->args_[3])) {
        collective_consumers.push_back({data_alloc, send_counts_alloc, op->span_});
      }
      return;
    }
  }

  void VisitExpr_(const CallPtr& op) override {
    AnalyzeDispatch(op);
    AnalyzeCollective(op);
    IRVisitor::VisitExpr_(op);
  }

  // A captured (`as tid`) chip-orch dispatch is a Submit; the ``device=`` attr
  // it carries is analysed identically through the Call-shaped view.
  void VisitExpr_(const SubmitPtr& op) override {
    AnalyzeDispatch(SubmitToCallView(op));
    IRVisitor::VisitExpr_(op);
  }

  std::vector<CollectiveConsumer> collective_consumers;

 private:
  [[nodiscard]] const WindowRecord* ResolveWindowRecord(const VarPtr& var) const {
    std::unordered_set<const Var*> visited;
    return ResolveWindowRecord(var, &visited);
  }

  [[nodiscard]] const WindowRecord* ResolveWindowRecord(const VarPtr& var,
                                                        std::unordered_set<const Var*>* visited) const {
    if (!var || !visited->insert(var.get()).second) return nullptr;
    auto direct = view_to_window_.find(var.get());
    if (direct != view_to_window_.end()) return &direct->second;

    // An SSA loop carry is an IterArg with no AssignStmt RHS: resolve through
    // its init value (AsVarLike + initValue_, mirroring InferTileMemorySpace
    // #2547 — see .claude/rules/ir-kind-traits.md).
    if (var->GetKind() == ObjectKind::IterArg) {
      auto init = static_cast<const IterArg*>(var.get())->initValue_;
      return init ? ResolveWindowRecord(AsVarLike(init), visited) : nullptr;
    }

    auto def_it = var_defs_.find(var.get());
    if (def_it == var_defs_.end() || !def_it->second) return nullptr;
    if (auto alias = AsVarLike(def_it->second)) {
      return ResolveWindowRecord(alias, visited);
    }
    auto call = As<Call>(def_it->second);
    if (call && IsTensorAllReduce(call) && !call->args_.empty()) {
      return ResolveWindowRecord(AsVarLike(call->args_[0]), visited);
    }
    if (call && IsTensorAllToAll(call) && call->args_.size() > 1) {
      return ResolveWindowRecord(AsVarLike(call->args_[1]), visited);
    }
    if (call && IsTensorAllGather(call) && call->args_.size() > 1) {
      return ResolveWindowRecord(AsVarLike(call->args_[1]), visited);
    }
    if (call && IsTensorAllToAllV(call) && call->args_.size() > 1) {
      return ResolveWindowRecord(AsVarLike(call->args_[1]), visited);
    }
    return nullptr;
  }

  const std::unordered_map<const Var*, WindowRecord>& view_to_window_;
  const std::map<std::string, FunctionPtr>& chip_orchs_;
  const std::unordered_map<const Var*, ExprPtr>& var_defs_;
  std::vector<ForStmtPtr> for_stack_;
  int repeating_scope_depth_ = 0;
};

/// A host-orchestration function in PyPTO is declared as either
/// ``@pl.function(type=FunctionType.Orchestration, level=Level.HOST)`` or
/// (more common in distributed programs) ``@pl.function(level=Level.HOST,
/// role=Role.Orchestrator)`` where ``func_type_`` may stay ``Opaque``. Accept
/// either form so the pass works with the conventional host_orch declaration
/// idiom used in distributed tests.
[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::HOST) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

[[nodiscard]] bool IsChipOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::CHIP) return false;
  // A Graph function carries {CHIP, Orchestrator} exactly like a chip entry, so
  // the role disjunct below matches it even though its func_type_ does not. It
  // is a task launched *by* a chip entry, never a host dispatch target, so
  // exclude it explicitly — narrowing the func_type_ term alone would not help.
  if (func->func_type_ == FunctionType::Graph) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

/// Build a fresh Var with an updated DistributedTensorType whose
/// ``window_buffer_`` now points to the constructed ``wb``.
[[nodiscard]] VarPtr MintViewVar(const VarPtr& old_var, const WindowBufferPtr& wb) {
  auto dt = As<DistributedTensorType>(old_var->GetType());
  INTERNAL_CHECK_SPAN(dt, old_var->span_)
      << "MaterializeCommDomainScopes: pld.tensor.window result Var should have DistributedTensorType";
  // Preserve every field (shape / dtype / memref / tensor_view) and set
  // window_buffer to the freshly-built ``wb``. ``pld.tensor.window`` outputs never
  // carry memref / tensor_view today (parser-fresh views), but the full-fields
  // ctor is the safe form.
  auto new_type = std::make_shared<const DistributedTensorType>(dt->shape_, dt->dtype_, dt->memref_,
                                                                dt->tensor_view_, std::make_optional(wb));
  return std::make_shared<Var>(old_var->name_hint_, new_type, old_var->span_);
}

/// Build a fresh IterArg with an updated DistributedTensorType whose
/// ``window_buffer_`` points to ``wb`` — a loop carry that views a window must
/// share the window-buffer object with its init value / yield / return var
/// (the typecheck verifier compares those by pointer identity).
[[nodiscard]] IterArgPtr MintIterArgWithWb(const IterArgPtr& iter_arg, const WindowBufferPtr& wb) {
  auto dt = As<DistributedTensorType>(iter_arg->GetType());
  INTERNAL_CHECK_SPAN(dt, iter_arg->span_)
      << "MaterializeCommDomainScopes: loop iter_arg should have DistributedTensorType";
  auto new_type = std::make_shared<const DistributedTensorType>(dt->shape_, dt->dtype_, dt->memref_,
                                                                dt->tensor_view_, std::make_optional(wb));
  return std::make_shared<IterArg>(iter_arg->name_hint_, new_type, iter_arg->initValue_, iter_arg->span_);
}

/// Re-mint the Call that *defines* a re-typed view Var so its own result type
/// carries the same ``window_buffer_`` back-reference.
///
/// ``transform_utils::Substitute`` rewrites Var *references* — including an
/// AssignStmt's LHS — but never the result type of the expression on the RHS.
/// Without this rewrite, every ``data = pld.tensor.window(...)`` ends the pass
/// with ``var.type.window_buffer_`` set and ``value.type.window_buffer_`` still
/// ``nullopt``, which violates the ``AssignTypeSymmetry`` contract (#1285) that
/// an AssignStmt's two sides carry structurally equal types.
///
/// The rewrite only ever *adds* the back-reference: it rebuilds the Call type
/// from the Call's own fields plus the Var's window buffer and keeps it only
/// when that reconstruction is structurally equal to the Var's type, so a
/// genuine shape / dtype / view disagreement is left alone to be reported
/// rather than papered over.
///
/// Call-only is exhaustive here, not an oversight of ``Submit`` (see
/// `pass-submit-awareness.md`): a Var enters ``view_subst`` only as a
/// ``pld.tensor.window`` result, a collective result, or a loop carry over one
/// of those, and the collector records all three from a ``Call`` RHS. A Submit
/// returns a TASK_ID-augmented TupleType, which is not a window view type.
class DefiningCallRetyper : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(base);
    if (!assign || !assign->var_ || !assign->value_) return base;

    auto var_dt = As<DistributedTensorType>(assign->var_->GetType());
    if (!var_dt || !var_dt->window_buffer_.has_value()) return base;

    auto call = As<Call>(assign->value_);
    if (!call) return base;
    auto call_dt = As<DistributedTensorType>(call->GetType());
    if (!call_dt || structural_equal(call_dt, var_dt)) return base;

    auto retyped = std::make_shared<const DistributedTensorType>(
        call_dt->shape_, call_dt->dtype_, call_dt->memref_, call_dt->tensor_view_, var_dt->window_buffer_);
    if (!structural_equal(CloneTypeWithoutMemRef(retyped), CloneTypeWithoutMemRef(var_dt))) return base;

    auto new_call =
        std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_, retyped, call->span_);
    return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
  }
};

/// Follow a Var's def chain (through window / allreduce / all_to_all / allgather /
/// all_to_all_v aliases and loop-carry init values) to the alloc-window-buffer it
/// views, returning that alloc's ``WindowBuffer`` (the SAME object the scope slot
/// holds). Mirrors DispatchAnalyzer::ResolveWindowRecord; used to extend the view
/// substitution to collective results and loop carries so every DistributedTensorType
/// denoting the same window shares one WindowBuffer object.
[[nodiscard]] const WindowBufferPtr* ResolveWindowBufferPtr(const VarPtr& var,
                                                            const AllocAndWindowCollector& c) {
  std::unordered_set<const Var*> visited;
  VarPtr cur = var;
  while (cur && visited.insert(cur.get()).second) {
    auto it = c.view_to_window.find(cur.get());
    if (it != c.view_to_window.end()) return &it->second.alloc->wb;
    if (cur->GetKind() == ObjectKind::IterArg) {
      auto init = static_cast<const IterArg*>(cur.get())->initValue_;
      cur = init ? AsVarLike(init) : nullptr;
      continue;
    }
    auto def_it = c.var_defs.find(cur.get());
    if (def_it == c.var_defs.end() || !def_it->second) return nullptr;
    const ExprPtr& def = def_it->second;
    if (auto alias = AsVarLike(def)) {
      cur = alias;
      continue;
    }
    auto call = As<Call>(def);
    if (call && IsOp(call, "pld.tensor.window") && !call->args_.empty()) {
      cur = AsVarLike(call->args_[0]);
      continue;
    }
    if (call && IsTensorAllReduce(call) && !call->args_.empty()) {
      cur = AsVarLike(call->args_[0]);
      continue;
    }
    if (call && (IsOp(call, "pld.tensor.all_to_all") || IsOp(call, "pld.tensor.allgather")) &&
        call->args_.size() > 1) {
      cur = AsVarLike(call->args_[1]);
      continue;
    }
    if (call && (IsOp(call, "pld.tensor.all_to_all_v") || IsOp(call, "pld.tensor.indexed_all_to_all_v")) &&
        call->args_.size() > 1) {
      cur = AsVarLike(call->args_[1]);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Process one host_orch function: identify allocs/windows/dispatches,
/// construct WindowBuffer instances, rewrite the body to substitute view Vars
/// with type-updated copies, and wrap the body in a chain of
/// ``CommDomainScopeStmt`` (one per inferred comm domain, outer = first
/// declared, inner = last declared).
FunctionPtr ProcessHostOrch(const FunctionPtr& func, const std::map<std::string, FunctionPtr>& chip_orchs,
                            const BufferEngines& engines) {
  // Idempotence: if this function's body is already wrapped in a
  // CommDomainScopeStmt, the pass has run on it before — skip to avoid
  // double-wrapping the body and minting a fresh set of WindowBuffer
  // instances that would shadow the existing ones on every view.
  if (func->body_ && As<CommDomainScopeStmt>(func->body_)) {
    // Preserve the WindowBuffer objects referenced by existing tensor types,
    // but split an old scope if newly introduced uses need distinct engines.
    class EngineUpdater : public IRMutator {
     public:
      explicit EngineUpdater(const BufferEngines& engines) : engines_(engines) {}
      std::set<std::string> names;
      StmtPtr VisitStmt_(const CommDomainScopeStmtPtr& op) override {
        auto updated = As<CommDomainScopeStmt>(IRMutator::VisitStmt_(op));
        std::array<bool, 2> combined{};
        for (const auto& slot : updated->slots_) {
          const auto uses = EnginesFor(engines_, slot->base_.get());
          combined[0] |= uses[0];
          combined[1] |= uses[1];
        }
        // Explicit materialized IR may declare a workspace for an opaque
        // kernel. With no recognized uses, retain that declaration unchanged.
        if (!combined[0] && !combined[1]) return updated;
        std::array<std::vector<WindowBufferPtr>, 2> groups;
        for (const auto& slot : updated->slots_) {
          const auto uses = EnginesFor(engines_, slot->base_.get());
          const bool urma = uses[0] || (!uses[1] && combined[0] && !combined[1]);
          groups[urma ? 1 : 0].push_back(slot);
        }
        auto engines = updated->engines_;
        engines.erase(std::remove(engines.begin(), engines.end(), "urma"), engines.end());
        const bool split = !groups[0].empty() && !groups[1].empty();
        if (!split) {
          if (!groups[1].empty()) engines.emplace_back("urma");
          if (engines == updated->engines_) return updated;
          auto result = MutableCopy(updated);
          result->engines_ = std::move(engines);
          return result;
        }
        // Keep the original name on the outer scope. New names are allocated
        // from one monotonic sequence, avoiding collisions with existing IR.
        std::string name;
        do {
          name = "comm_engine_" + std::to_string(next_name_++);
        } while (!names.insert(name).second);
        const auto first = EnginesFor(engines_, updated->slots_.front()->base_.get());
        const size_t outer_group = first[0] ? 1 : 0;
        const size_t inner_group = 1 - outer_group;
        auto inner = MutableCopy(updated);
        inner->name_hint_ = std::move(name);
        inner->slots_ = std::move(groups[inner_group]);
        inner->engines_ = engines;
        if (inner_group == 1) inner->engines_.emplace_back("urma");
        auto outer = MutableCopy(updated);
        outer->slots_ = std::move(groups[outer_group]);
        outer->engines_ = std::move(engines);
        if (outer_group == 1) outer->engines_.emplace_back("urma");
        outer->body_ = inner;
        return outer;
      }

     private:
      const BufferEngines& engines_;
      size_t next_name_ = 0;
    } updater(engines);
    class NameCollector : public IRVisitor {
     public:
      std::set<std::string> names;
      void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
        names.insert(op->name_hint_);
        IRVisitor::VisitStmt_(op);
      }
    } names;
    names.VisitStmt(func->body_);
    updater.names = std::move(names.names);
    auto body = updater.VisitStmt(func->body_);
    if (body == func->body_) return func;
    auto result = MutableCopy(func);
    result->body_ = std::move(body);
    return result;
  }

  StmtPtr materialization_body = func->body_;

  AllocAndWindowCollector collector;
  collector.VisitStmt(materialization_body);

  if (collector.allocs.empty()) {
    // No window-buffer allocations in this host_orch — nothing to do.
    return func;
  }

  // Phase 2: record device-descriptor evidence from dispatch sites.
  DispatchAnalyzer analyzer(collector.view_to_window, chip_orchs, collector.var_defs);
  analyzer.VisitStmt(materialization_body);

  // Host-level collectives do not carry their own device= selector. Their
  // signal buffer is a user-visible window slot, so inherit paired data/target
  // coverage when present. Barrier-only consumers keep signal coverage from
  // dispatch sites, or fall back to the full comm-domain device set.
  for (const auto& consumer : analyzer.collective_consumers) {
    INTERNAL_CHECK_SPAN(consumer.signal_alloc, consumer.span)
        << "MaterializeCommDomainScopes: invalid collective consumer bookkeeping";
    if (consumer.data_alloc) {
      INTERNAL_CHECK_SPAN(!consumer.data_alloc->seen.empty(), consumer.span)
          << "MaterializeCommDomainScopes: collective data/target buffer has no inferred comm-domain "
             "coverage to share with its signal buffer";
      consumer.signal_alloc->seen.insert(consumer.signal_alloc->seen.end(), consumer.data_alloc->seen.begin(),
                                         consumer.data_alloc->seen.end());
      continue;
    }
    // Barrier-only consumer: if no dispatch sites registered coverage for the
    // signal window, surface the missing coverage as a user-facing diagnostic
    // rather than silently widening to the full comm domain (which would cause
    // a cross-rank hang for device-subset barriers).
    CHECK_SPAN(!consumer.signal_alloc->seen.empty(), consumer.span)
        << "MaterializeCommDomainScopes: pld.tensor.barrier signal buffer has no inferred "
           "comm-domain coverage — add a device= annotation or ensure the signal window "
           "is consumed by a device-tagged chip dispatch";
  }

  // Phase 4: each alloc must have at least one window AND at least one
  // consuming dispatch — otherwise it is dead and downstream codegen has
  // nothing to point a CommDomain buffer slot at.
  std::unordered_map<const Var*, std::vector<const WindowRecord*>> allocs_with_windows;
  for (const auto& w : collector.windows) {
    allocs_with_windows[w.alloc->ptr_var.get()].push_back(&w);
  }
  for (const auto& rec : collector.allocs) {
    INTERNAL_CHECK_SPAN(!allocs_with_windows[rec->ptr_var.get()].empty(), rec->span)
        << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer '" << rec->name
        << "' has no pld.tensor.window materialisation (dead allocation)";
    INTERNAL_CHECK_SPAN(!rec->seen.empty(), rec->span)
        << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer '" << rec->name
        << "' is not consumed by any chip_orch dispatch";
  }

  // Phase 4: construct WindowBuffer for each alloc. Final descriptor merging
  // happens in Phase 6 below (per-group), so we don't precompute it here.
  for (auto& rec : collector.allocs) {
    rec->wb = std::make_shared<const WindowBuffer>(
        rec->ptr_var, rec->size_expr,
        /*load_from_host=*/false,
        /*store_to_host=*/false, rec->span, rec->alloc_call->GetKwarg<std::string>("memory_source", "device"),
        rec->alloc_call->GetKwarg<std::string>("init", ""));
  }

  // Every direct pld.tensor.window view must fit the logical size of its
  // allocation. Reject an oversized view even if the program only accesses a
  // prefix: the view itself must be valid independently of its consumers.
  for (const auto& w : collector.windows) {
    auto view_dt = As<DistributedTensorType>(w.old_view_var->GetType());
    INTERNAL_CHECK_SPAN(view_dt, w.old_view_var->span_)
        << "MaterializeCommDomainScopes: pld.tensor.window result should have DistributedTensorType";
    const auto fit = EvaluateWindowViewFit(*view_dt, *w.alloc->wb);
    CHECK_SPAN(fit.fit != WindowViewFit::kViolation,
               w.window_call ? w.window_call->span_ : w.old_view_var->span_)
        << "MaterializeCommDomainScopes: "
        << DescribeWindowViewFit(w.old_view_var->name_hint_, *w.alloc->wb, fit);
  }

  // Phase 5: build var substitution map for every pld.tensor.window result Var,
  // every allreduce result Var (aliases its data window), and every loop carry
  // (iter_arg / return_var) whose lineage is a window. The typecheck verifier
  // compares loop-carry types (initValue / iter_arg / yield / return_var) by
  // WindowBuffer pointer identity, so all DistributedTensorTypes denoting the
  // same window must share one WindowBuffer object after substitution.
  std::unordered_map<const Var*, VarPtr> view_subst;
  for (const auto& w : collector.windows) {
    view_subst[w.old_view_var.get()] = MintViewVar(w.old_view_var, w.alloc->wb);
  }
  for (const auto& [result_var, data_var] : collector.collective_results) {
    if (!result_var || !data_var) continue;
    if (const auto* wb = ResolveWindowBufferPtr(data_var, collector)) {
      if (view_subst.find(result_var.get()) == view_subst.end()) {
        view_subst[result_var.get()] = MintViewVar(result_var, *wb);
      }
    }
  }
  auto sync_loop_carries = [&collector, &view_subst](const std::vector<IterArgPtr>& iter_args,
                                                     const std::vector<VarPtr>& return_vars) {
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& ia = iter_args[i];
      if (!ia || !ia->initValue_) continue;
      const auto* wb = ResolveWindowBufferPtr(AsVarLike(ia->initValue_), collector);
      if (!wb) continue;
      if (view_subst.find(ia.get()) == view_subst.end()) {
        view_subst[ia.get()] = MintIterArgWithWb(ia, *wb);
      }
      if (i < return_vars.size() && return_vars[i] &&
          view_subst.find(return_vars[i].get()) == view_subst.end()) {
        view_subst[return_vars[i].get()] = MintViewVar(return_vars[i], *wb);
      }
    }
  };
  for (const auto& fs : collector.loops) sync_loop_carries(fs->iter_args_, fs->return_vars_);
  for (const auto& ws : collector.whiles) sync_loop_carries(ws->iter_args_, ws->return_vars_);

  // Phase 6: cluster allocs by merged device descriptor, memory source and
  // engine category (alloc-order within a domain). Use a vector for deterministic order: scan
  // collector.allocs in source order and append to the first matching entry
  // or create a new one. `desc_to_index` maps the composite grouping key to its
  // `pending` slot (O(log domains) via DeviceDescriptor::operator<, per
  // pass-complexity.md — a per-alloc linear scan of `pending` would make this
  // Phase O(allocs^2) for programs with many distinct comm domains). Store
  // indices, not pointers, into the map: `pending.push_back` may reallocate
  // and invalidate any previously-taken `PendingDomain*`.
  struct PendingDomain {
    DeviceDescriptor desc;
    std::vector<WindowBufferPtr> slots;
    std::set<std::string> names;  // sanity check
    Span span;
    bool uses_urma;
  };
  std::vector<PendingDomain> pending;
  using Coverage = std::pair<DeviceDescriptor, std::string>;
  std::map<Coverage, std::array<bool, 2>> coverage_engines;
  std::unordered_map<const Var*, Coverage> coverage;
  for (const auto& rec : collector.allocs) {
    DeviceDescriptor merged;
    for (const auto& d : rec->seen) merged.Merge(d);
    auto key = std::make_pair(merged, rec->wb->memory_source_);
    coverage.emplace(rec->ptr_var.get(), key);
    const auto uses = EnginesFor(engines, rec->ptr_var.get());
    auto& combined = coverage_engines[key];
    combined[0] |= uses[0];
    combined[1] |= uses[1];
  }
  std::map<std::tuple<DeviceDescriptor, std::string, bool>, size_t> desc_to_index;
  for (const auto& rec : collector.allocs) {
    const auto& [merged, memory_source] = coverage.at(rec->ptr_var.get());
    const auto uses = EnginesFor(engines, rec->ptr_var.get());
    const auto combined = coverage_engines.at(coverage.at(rec->ptr_var.get()));
    // Unclassified local-only slots may share a compatible domain. Prefer the
    // default MTE/no-workspace group when both explicit categories are present.
    const bool uses_urma = uses[0] || (!uses[1] && combined[0] && !combined[1]);
    const auto key = std::make_tuple(merged, memory_source, uses_urma);
    auto it = desc_to_index.find(key);
    size_t index;
    if (it == desc_to_index.end()) {
      index = pending.size();
      desc_to_index.emplace(key, index);
      pending.push_back({merged, {}, {}, rec->span, uses_urma});
    } else {
      index = it->second;
    }
    PendingDomain& tgt = pending[index];
    INTERNAL_CHECK_SPAN(tgt.names.insert(rec->name).second, rec->span)
        << "MaterializeCommDomainScopes: duplicate allocation name '" << rec->name
        << "' within the same comm domain";
    tgt.slots.push_back(rec->wb);
  }

  // Phase 7: rewrite host_orch body so every reference to a pld.tensor.window result
  // Var picks up the type-updated copy. The base IRMutator handles all uses;
  // Substitute is the wrapper that does exactly this transformation.
  StmtPtr new_body = materialization_body;
  if (!view_subst.empty()) {
    new_body = transform_utils::Substitute(new_body, view_subst);
    // Substitute rewrites references only; bring each re-typed Var's defining
    // Call along so both sides of the assignment agree (#1285).
    DefiningCallRetyper retyper;
    new_body = retyper.VisitStmt(new_body);
  }

  // Phase 8: wrap new_body in nested CommDomainScopeStmts. Outer = first
  // declared domain, inner = last. ``name_hint_`` is ``"comm_d<n>"`` so
  // DistributedCodegen emits the matching ``__comm_d<n>`` handle var.
  //
  // Build inner-out: start from the substituted body, wrap once per group
  // in reverse iteration order.
  for (size_t i = pending.size(); i-- > 0;) {
    auto& g = pending[i];
    std::string name_hint = "comm_d" + std::to_string(i);
    new_body = std::make_shared<const CommDomainScopeStmt>(
        g.desc.ToDevices(), std::move(g.slots), std::move(name_hint), new_body, g.span,
        std::vector<std::string>{}, std::vector<std::pair<std::string, std::any>>{},
        g.uses_urma ? std::vector<std::string>{"urma"} : std::vector<std::string>{});
  }

  if (new_body.get() == func->body_.get()) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass MaterializeCommDomainScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    const auto engines = BufferEngineCollector(program).Collect();
    // Index chip-level Orchestration functions by name so the dispatch
    // analyzer can recognise host → chip Calls.
    std::map<std::string, FunctionPtr> chip_orchs;
    for (const auto& [gv, func] : program->functions_) {
      if (IsChipOrch(func)) chip_orchs[func->name_] = func;
    }

    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool modified = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (!IsHostOrch(func)) {
        new_functions[gvar] = func;
        continue;
      }
      auto new_func = ProcessHostOrch(func, chip_orchs, engines);
      new_functions[gvar] = new_func;
      if (new_func.get() != func.get()) modified = true;
    }

    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "MaterializeCommDomainScopes", kMaterializeCommDomainScopesProperties);
}

}  // namespace pass

// ============================================================================
// CommDomainScopesMaterialized property verifier
//
// Walks each host_orch function body looking for ``CommDomainScopeStmt``s and
// asserts: (a) ``slots_`` is non-empty (an empty domain would emit
// ``window_size=0`` and the runtime would reject it); (b) every slot is
// non-null; (c) each ``WindowBuffer`` appears as a slot in at most one
// scope program-wide (cross-scope identity uniqueness).
// ============================================================================

class CommDomainScopeCollector : public IRVisitor {
 public:
  std::vector<CommDomainScopeStmtPtr> scopes;
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    scopes.push_back(op);
    if (op->body_) VisitStmt(op->body_);
  }
};

class CommDomainScopesMaterializedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "CommDomainScopesMaterialized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    CommDomainScopeCollector collector;
    for (const auto& [gvar, func] : program->functions_) {
      if (func && func->body_) collector.VisitStmt(func->body_);
    }
    std::unordered_map<const WindowBuffer*, const CommDomainScopeStmt*> first_seen;
    for (const auto& scope : collector.scopes) {
      if (!scope) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                 "null CommDomainScopeStmt in IR", Span::unknown());
        continue;
      }
      if (scope->slots_.empty()) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                 "CommDomainScopeStmt '" + scope->name_hint_ +
                                     "' has no slots — every comm-domain scope must carry at least one "
                                     "WindowBuffer",
                                 scope->span_);
        continue;
      }
      for (const auto& slot : scope->slots_) {
        if (!slot) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                   "CommDomainScopeStmt '" + scope->name_hint_ + "' has a null slot",
                                   scope->span_);
          continue;
        }
        auto [it, inserted] = first_seen.emplace(slot.get(), scope.get());
        if (!inserted) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                   "WindowBuffer '" + slot->name_hint_ +
                                       "' appears in multiple CommDomainScopeStmts ('" +
                                       it->second->name_hint_ + "' and '" + scope->name_hint_ + "')",
                                   slot->span_);
        }
      }
    }
  }
};

PropertyVerifierPtr CreateCommDomainScopesMaterializedPropertyVerifier() {
  return std::make_shared<CommDomainScopesMaterializedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
