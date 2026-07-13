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

#include <any>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

[[nodiscard]] bool IsTensorAllReduce(const CallPtr& call) {
  return call && call->op_ && IsOp(call, "pld.tensor.allreduce");
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
    auto var = As<Var>(op->var_);
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
        auto ptr_arg_var = As<Var>(call->args_[0]);
        if (ptr_arg_var) {
          auto it = ptr_to_alloc.find(ptr_arg_var.get());
          if (it != ptr_to_alloc.end()) {
            WindowRecord wr{call, var, it->second};
            view_to_window[var.get()] = wr;
            windows.push_back(wr);
          }
        }
      }
    }
    // Record the AssignStmt def for every Var so ResolveDeviceDescriptor can
    // follow ``for r in pl.range(<var>)`` back to ``<var> = pld.system.world_size()``
    // (CSE / NormalizeStmtStructure hoists such calls out into a temp).
    if (auto var = As<Var>(op->var_)) {
      var_defs[var.get()] = op->value_;
    }
    IRVisitor::VisitStmt_(op);
  }

  std::vector<std::unique_ptr<AllocRecord>> allocs;
  std::unordered_map<const Var*, AllocRecord*> ptr_to_alloc;
  std::unordered_map<const Var*, WindowRecord> view_to_window;
  std::vector<WindowRecord> windows;
  std::unordered_map<const Var*, ExprPtr> var_defs;
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
    IRVisitor::VisitStmt_(op);
    for_stack_.pop_back();
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
      auto arg_var = As<Var>(arg);
      if (!arg_var) continue;
      auto window = ResolveWindowRecord(arg_var);
      if (window) {
        window->alloc->seen.push_back(desc);
      }
    }
  }

  [[nodiscard]] AllocRecord* ResolveWindowAlloc(const ExprPtr& expr, const std::string& op_name,
                                                const char* role) {
    auto view_var = As<Var>(expr);
    INTERNAL_CHECK_SPAN(view_var, expr->span_)
        << "MaterializeCommDomainScopes: " << op_name << " " << role << " must be a window view Var";
    auto window = ResolveWindowRecord(view_var);
    INTERNAL_CHECK_SPAN(window, expr->span_) << "MaterializeCommDomainScopes: " << op_name << " " << role
                                             << " must be produced by pld.tensor.window";
    return window->alloc;
  }

  void AnalyzeCollective(const CallPtr& op) {
    if (!op || !op->op_) return;

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
      if (op->args_.size() == 2) {
        collective_consumers.push_back({ResolveWindowAlloc(op->args_[0], "pld.tensor.allgather", "target"),
                                        ResolveWindowAlloc(op->args_[1], "pld.tensor.allgather", "signal"),
                                        op->span_});
        return;
      }
      INTERNAL_CHECK_SPAN(op->args_.size() == 4, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.allgather expects 2 args (host builtin) or "
             "4 args (InCore composite)";
      collective_consumers.push_back({ResolveWindowAlloc(op->args_[1], "pld.tensor.allgather", "target"),
                                      ResolveWindowAlloc(op->args_[2], "pld.tensor.allgather", "signal"),
                                      op->span_});
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

    auto def_it = var_defs_.find(var.get());
    if (def_it == var_defs_.end() || !def_it->second) return nullptr;
    if (auto alias = As<Var>(def_it->second)) {
      return ResolveWindowRecord(alias, visited);
    }
    auto call = As<Call>(def_it->second);
    if (!IsTensorAllReduce(call) || call->args_.empty()) return nullptr;
    return ResolveWindowRecord(As<Var>(call->args_[0]), visited);
  }

  const std::unordered_map<const Var*, WindowRecord>& view_to_window_;
  const std::map<std::string, FunctionPtr>& chip_orchs_;
  const std::unordered_map<const Var*, ExprPtr>& var_defs_;
  std::vector<ForStmtPtr> for_stack_;
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

/// Process one host_orch function: identify allocs/windows/dispatches,
/// construct WindowBuffer instances, rewrite the body to substitute view Vars
/// with type-updated copies, and wrap the body in a chain of
/// ``CommDomainScopeStmt`` (one per inferred comm domain, outer = first
/// declared, inner = last declared).
FunctionPtr ProcessHostOrch(const FunctionPtr& func, const std::map<std::string, FunctionPtr>& chip_orchs) {
  // Idempotence: if this function's body is already wrapped in a
  // CommDomainScopeStmt, the pass has run on it before — skip to avoid
  // double-wrapping the body and minting a fresh set of WindowBuffer
  // instances that would shadow the existing ones on every view.
  if (func->body_ && As<CommDomainScopeStmt>(func->body_)) {
    return func;
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

  // Phase 3: each alloc must have at least one window AND at least one
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
    rec->wb = std::make_shared<const WindowBuffer>(rec->ptr_var, rec->size_expr,
                                                   /*load_from_host=*/false,
                                                   /*store_to_host=*/false, rec->span);
  }

  // Phase 5: build var substitution map for every pld.tensor.window result Var.
  std::unordered_map<const Var*, VarPtr> view_subst;
  for (const auto& w : collector.windows) {
    view_subst[w.old_view_var.get()] = MintViewVar(w.old_view_var, w.alloc->wb);
  }

  // Phase 6: cluster allocs into pending domain entries by merged descriptor
  // (alloc-order within a domain). Use a vector for deterministic order: scan
  // collector.allocs in source order and append to the first matching entry
  // or create a new one.
  struct PendingDomain {
    DeviceDescriptor desc;
    std::vector<WindowBufferPtr> slots;
    std::set<std::string> names;  // sanity check
    Span span;
  };
  std::vector<PendingDomain> pending;
  for (const auto& rec : collector.allocs) {
    DeviceDescriptor merged;
    for (const auto& d : rec->seen) merged.Merge(d);
    PendingDomain* tgt = nullptr;
    for (auto& g : pending) {
      if (g.desc == merged) {
        tgt = &g;
        break;
      }
    }
    if (!tgt) {
      pending.push_back({merged, {}, {}, rec->span});
      tgt = &pending.back();
    }
    INTERNAL_CHECK_SPAN(tgt->names.insert(rec->name).second, rec->span)
        << "MaterializeCommDomainScopes: duplicate allocation name '" << rec->name
        << "' within the same comm domain";
    tgt->slots.push_back(rec->wb);
  }

  // Phase 7: rewrite host_orch body so every reference to a pld.tensor.window result
  // Var picks up the type-updated copy. The base IRMutator handles all uses;
  // Substitute is the wrapper that does exactly this transformation.
  StmtPtr new_body = view_subst.empty() ? materialization_body
                                        : transform_utils::Substitute(materialization_body, view_subst);

  // Phase 8: wrap new_body in nested CommDomainScopeStmts. Outer = first
  // declared domain, inner = last. ``name_hint_`` is ``"comm_d<n>"`` so
  // DistributedCodegen emits the matching ``__comm_d<n>`` handle var.
  //
  // Build inner-out: start from the substituted body, wrap once per group
  // in reverse iteration order.
  for (size_t i = pending.size(); i-- > 0;) {
    auto& g = pending[i];
    std::string name_hint = "comm_d" + std::to_string(i);
    new_body = std::make_shared<const CommDomainScopeStmt>(g.desc.ToDevices(), std::move(g.slots),
                                                           std::move(name_hint), new_body, g.span);
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
      auto new_func = ProcessHostOrch(func, chip_orchs);
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
