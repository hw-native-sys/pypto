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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

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
  CallPtr alloc_call;                  ///< pld.alloc_window_buffer Call
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

/// Per-pld.window result Var: maps the LHS Var pointer back to its alloc.
struct WindowRecord {
  CallPtr window_call;
  VarPtr old_view_var;
  AllocRecord* alloc;
};

/// Scans a host_orch function body once and records every
/// ``pld.alloc_window_buffer`` and ``pld.window`` assignment.
class AllocAndWindowCollector : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    auto var = As<Var>(op->var_);
    auto call = As<Call>(op->value_);
    if (var && call && call->op_) {
      const auto& op_name = call->op_->name_;
      if (op_name == "pld.alloc_window_buffer") {
        INTERNAL_CHECK_SPAN(call->args_.size() == 1, call->span_)
            << "CollectCommGroups: pld.alloc_window_buffer expects exactly one arg (size)";
        // The parser injects ``name`` as a kwarg derived from the assignment
        // LHS — not as an ``attrs`` entry — so use GetKwarg here.
        auto name = call->GetKwarg<std::string>("name");
        INTERNAL_CHECK_SPAN(!name.empty(), call->span_)
            << "CollectCommGroups: pld.alloc_window_buffer missing 'name' kwarg";
        auto rec = std::make_unique<AllocRecord>(call, var, call->args_[0], name, call->span_);
        ptr_to_alloc[var.get()] = rec.get();
        allocs.push_back(std::move(rec));
      } else if (op_name == "pld.window" && !call->args_.empty()) {
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
    IRVisitor::VisitStmt_(op);
  }

  std::vector<std::unique_ptr<AllocRecord>> allocs;
  std::unordered_map<const Var*, AllocRecord*> ptr_to_alloc;
  std::unordered_map<const Var*, WindowRecord> view_to_window;
  std::vector<WindowRecord> windows;
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

/// Resolves the device descriptor for a ``device=`` Expr in the context of a
/// stack of enclosing ForStmt scopes. Throws pypto::ValueError on unsupported
/// forms (the user's parser is meant to restrict ``device=`` to ConstInt or
/// the induction var of an enclosing pl.range loop).
DeviceDescriptor ResolveDeviceDescriptor(const ExprPtr& device, const std::vector<ForStmtPtr>& for_stack,
                                         const Span& dispatch_span) {
  DeviceDescriptor desc;
  if (auto ci = As<ConstInt>(device)) {
    CHECK(ci->value_ >= 0) << "CollectCommGroups: device= ConstInt must be non-negative, got " << ci->value_;
    desc.subset.insert(ci->value_);
    return desc;
  }
  if (auto v = As<Var>(device)) {
    for (auto it = for_stack.rbegin(); it != for_stack.rend(); ++it) {
      const auto& fs = *it;
      if (fs->loop_var_.get() == v.get()) {
        // Loop bound determines coverage.
        if (auto stop_call = As<Call>(fs->stop_)) {
          if (stop_call->op_ && stop_call->op_->name_ == "pld.world_size") {
            desc.is_all = true;
            return desc;
          }
        }
        if (auto stop_ci = As<ConstInt>(fs->stop_)) {
          int64_t start = 0;
          if (auto start_ci = As<ConstInt>(fs->start_)) start = start_ci->value_;
          int64_t step = 1;
          if (auto step_ci = As<ConstInt>(fs->step_)) step = step_ci->value_;
          CHECK(step == 1) << "CollectCommGroups: device=r over a non-unit-step loop is not supported "
                              "(step="
                           << step << ")";
          CHECK(start >= 0 && stop_ci->value_ >= start)
              << "CollectCommGroups: device=r loop range must be [0, N) with N>=0";
          for (int64_t i = start; i < stop_ci->value_; ++i) desc.subset.insert(i);
          return desc;
        }
        throw pypto::ValueError(
            "CollectCommGroups: device=r loop bound must be ConstInt or pld.world_size()");
      }
    }
    throw pypto::ValueError(
        "CollectCommGroups: device= Var is not the induction variable of any enclosing pl.range loop");
  }
  throw pypto::ValueError(
      "CollectCommGroups: device= expression must be ConstInt or the induction var of pl.range; got "
      "an unsupported expression at " +
      dispatch_span.to_string());
}

/// Walks a host_orch body, maintaining a stack of enclosing ForStmts, and for
/// every chip_orch dispatch Call records the inferred device descriptor against
/// each view Var passed positionally.
class DispatchAnalyzer : public IRVisitor {
 public:
  DispatchAnalyzer(const std::unordered_map<const Var*, WindowRecord>& view_to_window,
                   const std::map<std::string, FunctionPtr>& chip_orchs)
      : view_to_window_(view_to_window), chip_orchs_(chip_orchs) {}

  void VisitStmt_(const ForStmtPtr& op) override {
    for_stack_.push_back(op);
    IRVisitor::VisitStmt_(op);
    for_stack_.pop_back();
  }

  void VisitExpr_(const CallPtr& op) override {
    if (IsChipOrchDispatch(op, chip_orchs_)) {
      ExprPtr device;
      for (const auto& [k, v] : op->attrs_) {
        if (k == "device") {
          // attrs["device"] is stored as ExprPtr by N3 parser.
          if (const auto* p = std::any_cast<ExprPtr>(&v)) device = *p;
          break;
        }
      }
      if (device) {
        DeviceDescriptor desc = ResolveDeviceDescriptor(device, for_stack_, op->span_);
        for (const auto& arg : op->args_) {
          auto arg_var = As<Var>(arg);
          if (!arg_var) continue;
          auto it = view_to_window_.find(arg_var.get());
          if (it != view_to_window_.end()) {
            it->second.alloc->seen.push_back(desc);
          }
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  const std::unordered_map<const Var*, WindowRecord>& view_to_window_;
  const std::map<std::string, FunctionPtr>& chip_orchs_;
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
      << "CollectCommGroups: pld.window result Var should have DistributedTensorType";
  // Preserve every field (shape / dtype / memref / tensor_view) and set
  // window_buffer to the freshly-built ``wb``. ``pld.window`` outputs never
  // carry memref / tensor_view today (parser-fresh views), but the full-fields
  // ctor is the safe form.
  auto new_type = std::make_shared<const DistributedTensorType>(dt->shape_, dt->dtype_, dt->memref_,
                                                                dt->tensor_view_, std::make_optional(wb));
  return std::make_shared<Var>(old_var->name_hint_, new_type, old_var->span_);
}

/// Process one host_orch function: identify allocs/windows/dispatches,
/// construct WindowBuffer instances, rewrite the body to substitute view Vars
/// with type-updated copies. Appends newly-built CommGroups to ``groups``.
FunctionPtr ProcessHostOrch(const FunctionPtr& func, const std::map<std::string, FunctionPtr>& chip_orchs,
                            std::vector<CommGroupPtr>& groups) {
  AllocAndWindowCollector collector;
  collector.VisitStmt(func->body_);

  if (collector.allocs.empty()) {
    // No window-buffer allocations in this host_orch — nothing to do.
    return func;
  }

  // Phase 2: record device-descriptor evidence from dispatch sites.
  DispatchAnalyzer analyzer(collector.view_to_window, chip_orchs);
  analyzer.VisitStmt(func->body_);

  // Phase 3: each alloc must have at least one window AND at least one
  // consuming dispatch — otherwise it is dead and downstream codegen has
  // nothing to point ChipBootstrapConfig at.
  std::unordered_map<const Var*, std::vector<const WindowRecord*>> allocs_with_windows;
  for (const auto& w : collector.windows) {
    allocs_with_windows[w.alloc->ptr_var.get()].push_back(&w);
  }
  for (const auto& rec : collector.allocs) {
    CHECK(!allocs_with_windows[rec->ptr_var.get()].empty())
        << "CollectCommGroups: pld.alloc_window_buffer '" << rec->name
        << "' has no pld.window materialisation (dead allocation) at " << rec->span.to_string();
    CHECK(!rec->seen.empty()) << "CollectCommGroups: pld.alloc_window_buffer '" << rec->name
                              << "' is not consumed by any chip_orch dispatch at " << rec->span.to_string();
  }

  // Phase 4: construct WindowBuffer for each alloc. Final descriptor merging
  // happens in Phase 6 below (per-group), so we don't precompute it here.
  for (auto& rec : collector.allocs) {
    rec->wb = std::make_shared<const WindowBuffer>(rec->ptr_var, rec->size_expr,
                                                   /*load_from_host=*/false,
                                                   /*store_to_host=*/false, rec->span);
  }

  // Phase 5: build var substitution map for every pld.window result Var.
  std::unordered_map<const Var*, VarPtr> view_subst;
  for (const auto& w : collector.windows) {
    view_subst[w.old_view_var.get()] = MintViewVar(w.old_view_var, w.alloc->wb);
  }

  // Phase 6: cluster allocs into CommGroups by merged descriptor (alloc-order
  // within a group). Use a vector for deterministic group order: scan
  // collector.allocs in source order and append to the first matching group
  // or create a new one.
  struct PendingGroup {
    DeviceDescriptor desc;
    std::vector<WindowBufferPtr> slots;
    std::set<std::string> names;  // sanity check
    Span span;
  };
  std::vector<PendingGroup> pending;
  for (const auto& rec : collector.allocs) {
    DeviceDescriptor merged;
    for (const auto& d : rec->seen) merged.Merge(d);
    PendingGroup* tgt = nullptr;
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
        << "CollectCommGroups: duplicate allocation name '" << rec->name << "' within the same CommGroup";
    tgt->slots.push_back(rec->wb);
  }
  for (auto& g : pending) {
    groups.push_back(std::make_shared<const CommGroup>(g.desc.ToDevices(), std::move(g.slots), g.span));
  }

  // Phase 7: rewrite host_orch body so every reference to a pld.window result
  // Var picks up the type-updated copy. The base IRMutator handles all uses;
  // Substitute is the wrapper that does exactly this transformation.
  if (view_subst.empty()) return func;
  auto new_body = transform_utils::Substitute(func->body_, view_subst);
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass CollectCommGroups() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Index chip-level Orchestration functions by name so the dispatch
    // analyzer can recognise host → chip Calls.
    std::map<std::string, FunctionPtr> chip_orchs;
    for (const auto& [gv, func] : program->functions_) {
      if (IsChipOrch(func)) chip_orchs[func->name_] = func;
    }

    std::vector<CommGroupPtr> all_groups;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool modified = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (!IsHostOrch(func)) {
        new_functions[gvar] = func;
        continue;
      }
      auto new_func = ProcessHostOrch(func, chip_orchs, all_groups);
      new_functions[gvar] = new_func;
      if (new_func.get() != func.get()) modified = true;
    }

    if (!modified && all_groups.empty()) return program;
    return std::make_shared<Program>(std::move(new_functions), std::move(all_groups), program->name_,
                                     program->span_);
  };

  return CreateProgramPass(pass_func, "CollectCommGroups", kCollectCommGroupsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
