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

// HardSyncallOccupancyValid verifier (issue #1935).
//
// The hard (FFTS) form of `system.syncall` waits for *every* physical core of
// its `core_type` to arrive at the barrier. Completing it needs two independent
// guarantees from the enclosing SPMD launch:
//   1. Full occupancy — one block per physical core. If the launch does not
//      fill all those cores, the unlaunched cores never reach the barrier.
//   2. sync_start — all blocks co-resident at once. Even at full occupancy, a
//      non-sync_start launch may dispatch blocks in waves; an early wave hits
//      the FFTS barrier and waits for a later wave that cannot dispatch until
//      the early one retires, so the barrier deadlocks.
// Either gap leaves the FFTS wait unable to complete, the AICore times out on
// device (507018), and the core needs a reset. The soft (GM-polling) form
// exists for partial occupancy. This verifier turns both runtime footguns into
// compile-time errors.
//
// It runs after ExpandMixedKernel, where each launched kernel's FunctionType is
// resolved (AIV / AIC / Group), which is what lets us map SPMD blocks to physical
// cores per launch shape. It covers every SPMD launch site whose block count it
// can classify (see below):
//   - `FunctionType::Spmd` function  (scope-based `with/for pl.spmd`)          [core_num attr]
//   - `FunctionType::Group` function with a core_num attr (`pl.cluster()`-nested
//     `pl.spmd`, unwrapped by OutlineClusterScopes)                            [core_num attr]
//   - a `Submit` node with a core_num (`pl.spmd_submit(..., core_num=N)`)      [Submit::core_num_]
//
// A launch width is one of:
//   - a compile-time literal N, checked against the target SoC's physical counts;
//   - a *launch-shape query* — `pl.system.available_cluster_count()` /
//     `pl.system.available_aiv_count()`, directly or through the Var it binds.
//     These resolve on device to the run's own geometry, so they fill their
//     core type by construction and need no count comparison. They are the
//     portable spelling: a literal that matches this SoC's static counts still
//     under-fills a run that lands on a device with fewer usable cores;
//   - anything else (a host scalar, arithmetic) — not classifiable, not checked.
//
// Given a launch site's width and its direct callee kernel K:
//   - K standalone AIV  + aiv_only  -> #VECTOR cores, queried by available_aiv_count
//   - K standalone AIC  + aic_only  -> #CUBE cores,   queried by available_cluster_count
//   - K standalone AIV/AIC + a *different* core_type (incl. the default `mix`) ->
//     unsatisfiable: the other core type has zero participants in a single-core
//     launch, so the barrier can never complete regardless of the width.
//   - K Group (mixed AIC+AIV kernel) -> #CUBE core-groups (1 AIC each), queried by
//     available_cluster_count, for any barrier core_type; the hard syncall lives
//     in the AIC/AIV sub-kernels (a `mix` barrier is duplicated into both lanes
//     -> reported once).
// A literal N != required is an error (partial *and* over-occupancy), as is a
// query for the other core type. Independently, a launch that is exactly full
// but lacks sync_start is also an error.
//
// A bare hard-syncall kernel with no SPMD launch is not checked: occupancy is a
// launch-time property.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Collects every Call and Submit reachable within a single function body
// (recursing through nested statements, but never into other functions).
class CallSubmitCollector : public IRVisitor {
 public:
  std::vector<CallPtr> calls;
  std::vector<SubmitPtr> submits;

  void VisitExpr_(const CallPtr& op) override {
    if (op) calls.push_back(op);
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) submits.push_back(op);
    IRVisitor::VisitExpr_(op);
  }
};

// Resolve the distinct functions directly called by `fn` (callees only — op
// calls such as tile.load resolve to nullptr via GetFunction and are dropped).
// Deduplicated so a callee invoked more than once is checked (and reported) once.
// Handles both Call (scope-based launch) and Submit (spmd_submit), per
// pass-submit-awareness.
std::vector<FunctionPtr> DirectCallees(const FunctionPtr& fn, const ProgramPtr& program) {
  CallSubmitCollector collector;
  if (fn->body_) collector.VisitStmt(fn->body_);
  std::vector<FunctionPtr> out;
  std::unordered_set<const Function*> seen;
  auto add = [&](const OpPtr& op) {
    if (!op) return;
    auto callee = program->GetFunction(op->name_);
    if (callee && seen.insert(callee.get()).second) out.push_back(callee);
  };
  for (const auto& call : collector.calls) add(call->op_);
  for (const auto& submit : collector.submits) add(submit->op_);
  return out;
}

struct HardSyncall {
  std::string core_type;
  Span span;
};

// How a launch site spells its block count. `Const` carries a compile-time
// literal; the two query kinds name a `pl.system.available_*_count()` call,
// which resolves on device to that core type's own count and therefore fills
// it by construction; `Unknown` is anything else and is left unchecked.
enum class LaunchWidthKind { Const, ClusterCount, AivCount, Unknown };

struct LaunchWidth {
  LaunchWidthKind kind = LaunchWidthKind::Unknown;
  int64_t value = 0;  ///< Block count; only meaningful when kind == Const.
};

// The query op that resolves to a core type's own count, as DSL spelling.
std::string QueryName(LaunchWidthKind kind) {
  INTERNAL_CHECK(kind == LaunchWidthKind::AivCount || kind == LaunchWidthKind::ClusterCount)
      << "Internal error: QueryName expects a launch-shape query kind";
  return kind == LaunchWidthKind::AivCount ? "pl.system.available_aiv_count()"
                                           : "pl.system.available_cluster_count()";
}

// Classify a Call used as a launch width. Only the two launch-shape queries
// are recognised; every other op is an opaque scalar producer.
LaunchWidthKind QueryKindOf(const CallPtr& call) {
  if (!call || !call->op_) return LaunchWidthKind::Unknown;
  if (IsOp(call, "system.available_cluster_count")) return LaunchWidthKind::ClusterCount;
  if (IsOp(call, "system.available_aiv_count")) return LaunchWidthKind::AivCount;
  return LaunchWidthKind::Unknown;
}

// Program-wide map from an SSA Var to the launch-shape query that defines it.
// A launch site's core_num typically references a Var bound in the *caller*
// (`n = pl.system.available_cluster_count()` in the orchestration body, then
// `with pl.spmd(n)`), so resolution cannot be function-local. Built once per
// Verify() run — one pass over every function body, O(N log N) overall.
class LaunchQueryDefCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, LaunchWidthKind> var_to_query;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) {
      const LaunchWidthKind kind = QueryKindOf(As<Call>(op->value_));
      if (kind != LaunchWidthKind::Unknown) var_to_query.emplace(op->var_.get(), kind);
    }
    IRVisitor::VisitStmt_(op);
  }
};

// Collect the hard-form `system.syncall` calls in a kernel body. The hard form
// carries no `mode` kwarg (defaults to "hard"); the soft form sets mode="soft".
std::vector<HardSyncall> CollectHardSyncalls(const FunctionPtr& fn) {
  CallSubmitCollector collector;
  if (fn->body_) collector.VisitStmt(fn->body_);
  std::vector<HardSyncall> out;
  for (const auto& call : collector.calls) {
    if (!call->op_ || !IsOp(call, "system.syncall")) continue;
    if (call->GetKwarg<std::string>("mode", "hard") == "soft") continue;
    out.push_back({call->GetKwarg<std::string>("core_type", "mix"), call->span_});
  }
  return out;
}

std::string OccupancyMessage(const std::string& detail, int64_t launched, const std::string& requirement) {
  return "hard pl.system.syncall" + detail + " requires the SPMD launch to " + requirement +
         ", but the launch has " + std::to_string(launched) +
         " blocks. Use mode=\"soft\" (GM-polling) for partial occupancy.";
}

// Full occupancy fills every physical core, but the FFTS barrier also needs all
// those blocks *co-resident* — every core must be running this kernel at the
// same time. Only a sync_start launch guarantees that; without it the runtime
// may dispatch blocks in waves, so an early wave hits the barrier and waits for
// a later wave that cannot dispatch until the early one retires -> deadlock
// (507018). Occupancy is necessary but not sufficient; sync_start is the
// co-residency guarantee.
std::string SyncStartMessage(const std::string& detail) {
  return "hard pl.system.syncall" + detail +
         " requires the SPMD launch to set sync_start=True so all blocks are co-resident at the "
         "FFTS barrier; without it the runtime may dispatch blocks in waves and the barrier "
         "deadlocks on device (507018). Add sync_start=True to the launch, or use mode=\"soft\" "
         "(GM-polling).";
}

// A launch sized by the *other* core type's query fills that type, not the one
// the barrier waits on — full occupancy for the wrong set of cores.
std::string WrongQueryMessage(const std::string& detail, LaunchWidthKind used, LaunchWidthKind expected,
                              const std::string& requirement) {
  return "hard pl.system.syncall" + detail + " requires the SPMD launch to " + requirement +
         ", but the launch sizes itself on " + QueryName(used) + ". Use " + QueryName(expected) +
         " instead, or mode=\"soft\" (GM-polling) for partial occupancy.";
}

// Ordered launch-level gate for a hard syncall: full occupancy first, then
// co-residency (sync_start). Returns the diagnostic for the first unmet
// requirement, or nullopt when the launch satisfies both. Keeps the "occupancy
// before sync_start" policy in one place for the AIV / AIC / Group cases.
// `required` is the physical unit count of the barrier's core type and
// `matching_query` the launch-shape query that resolves to it; `requirement`
// names the occupancy target (e.g. "fill all 48 AIV cores ...").
std::optional<std::string> OccupancyGateMessage(const LaunchWidth& width, LaunchWidthKind matching_query,
                                                int64_t required, bool sync_start, const std::string& detail,
                                                const std::string& requirement) {
  if (width.kind == LaunchWidthKind::Const) {
    if (width.value != required) return OccupancyMessage(detail, width.value, requirement);
  } else if (width.kind != matching_query) {
    return WrongQueryMessage(detail, width.kind, matching_query, requirement);
  }
  if (!sync_start) return SyncStartMessage(detail);
  return std::nullopt;
}

std::string UnsatisfiableMessage(const std::string& core_type, const std::string& launch_kind,
                                 const std::string& missing_cores) {
  return "hard pl.system.syncall(core_type=\"" + core_type + "\") can never complete in a " + launch_kind +
         " launch: it waits for all " + missing_cores +
         " cores, but a single-core-type SPMD launch provides none of them. Use a barrier core_type that "
         "matches the launch, or launch this barrier from a mixed (cube+vector) kernel.";
}

}  // namespace

class HardSyncallOccupancyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HardSyncallOccupancy"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    // Occupancy depends on the target SoC's physical core counts; without a
    // configured backend (e.g. pure-IR unit tests) there is nothing to check.
    if (!backend::BackendConfig::IsConfigured()) return;
    const auto* be = backend::GetBackend();
    const int total_vector = be->GetCoreCount(ir::CoreType::VECTOR);
    const int total_cube = be->GetCoreCount(ir::CoreType::CUBE);

    // A launch width spelled as a query usually reaches the launch site through
    // a Var bound in the caller, so resolve those bindings program-wide first.
    LaunchQueryDefCollector query_defs;
    for (const auto& [gv, fn] : program->functions_) {
      if (fn && fn->body_) query_defs.VisitStmt(fn->body_);
    }

    for (const auto& [gv, fn] : program->functions_) {
      if (!fn) continue;

      // Launch-site functions: a Spmd wrapper (scope-based pl.spmd), or a Group
      // carrying a core_num attr (a pl.cluster()-nested pl.spmd unwrapped by
      // OutlineClusterScopes). A Group *without* core_num is a mixed-kernel
      // wrapper (a launch callee), handled inside CheckLaunchedKernel.
      const bool is_launch_fn = fn->func_type_ == FunctionType::Spmd ||
                                (fn->func_type_ == FunctionType::Group && fn->HasAttr("core_num"));
      if (is_launch_fn) {
        // sync_start rides as a launch-function attr, emitted only when true by
        // OutlineHierarchyScopes / OutlineClusterScopes (absent => false).
        CheckLaunchSite(fn->GetAttr<ExprPtr>("core_num", nullptr), fn->GetAttr<bool>("sync_start", false),
                        DirectCallees(fn, program), total_vector, total_cube, program, query_defs,
                        diagnostics);
      }

      // Submit launch sites: pl.spmd_submit(..., core_num=N) carries the block
      // count on the Submit node itself (a plain pl.submit leaves core_num_ unset).
      if (!fn->body_) continue;
      CallSubmitCollector collector;
      collector.VisitStmt(fn->body_);
      for (const auto& submit : collector.submits) {
        if (!submit->core_num_.has_value() || !submit->op_) continue;
        auto callee = program->GetFunction(submit->op_->name_);
        if (!callee) continue;
        CheckLaunchSite(*submit->core_num_, submit->sync_start_, {callee}, total_vector, total_cube, program,
                        query_defs, diagnostics);
      }
    }
  }

 private:
  // Classify a launch site's core_num: a literal, one of the launch-shape
  // queries (used directly or through the Var it binds), or unclassifiable.
  static LaunchWidth ClassifyWidth(const ExprPtr& core_num_expr, const LaunchQueryDefCollector& query_defs) {
    if (auto core_num_const = As<ConstInt>(core_num_expr)) {
      return {LaunchWidthKind::Const, core_num_const->value_};
    }
    if (const LaunchWidthKind kind = QueryKindOf(As<Call>(core_num_expr)); kind != LaunchWidthKind::Unknown) {
      return {kind, 0};
    }
    if (auto var = AsVarLike(core_num_expr)) {
      auto it = query_defs.var_to_query.find(var.get());
      if (it != query_defs.var_to_query.end()) return {it->second, 0};
    }
    return {LaunchWidthKind::Unknown, 0};
  }

  static void CheckLaunchSite(const ExprPtr& core_num_expr, bool sync_start,
                              const std::vector<FunctionPtr>& callees, int total_vector, int total_cube,
                              const ProgramPtr& program, const LaunchQueryDefCollector& query_defs,
                              std::vector<Diagnostic>& diagnostics) {
    if (!core_num_expr) return;
    const LaunchWidth width = ClassifyWidth(core_num_expr, query_defs);
    if (width.kind == LaunchWidthKind::Unknown) return;  // opaque launch count — cannot check statically
    for (const auto& callee : callees) {
      CheckLaunchedKernel(callee, width, sync_start, total_vector, total_cube, program, diagnostics);
    }
  }

  static void CheckLaunchedKernel(const FunctionPtr& kernel, const LaunchWidth& width, bool sync_start,
                                  int total_vector, int total_cube, const ProgramPtr& program,
                                  std::vector<Diagnostic>& diagnostics) {
    if (!kernel) return;

    // Standalone AIV kernel: 1 block = 1 AIV core. Only an aiv_only barrier is
    // satisfiable; a mix / aic_only barrier waits for CUBE cores that no AIV-only
    // launch provides. (A dual-AIV-dispatched AIV kernel is a mixed-kernel lane
    // reached via Group, handled by the Group branch — not a standalone launch.)
    if (kernel->func_type_ == FunctionType::AIV && !kernel->HasAttr("dual_aiv_dispatch")) {
      for (const auto& hs : CollectHardSyncalls(kernel)) {
        if (hs.core_type != "aiv_only") {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0,
                                   UnsatisfiableMessage(hs.core_type, "vector-only (AIV)", "CUBE"), hs.span);
        } else if (auto msg = OccupancyGateMessage(width, LaunchWidthKind::AivCount, total_vector, sync_start,
                                                   "(core_type=\"aiv_only\")",
                                                   "fill all " + std::to_string(total_vector) +
                                                       " AIV cores (one block per physical core)")) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0, *msg, hs.span);
        }
      }
      return;
    }

    // Standalone AIC kernel: 1 block = 1 AIC core. Symmetric to the AIV case.
    if (kernel->func_type_ == FunctionType::AIC) {
      for (const auto& hs : CollectHardSyncalls(kernel)) {
        if (hs.core_type != "aic_only") {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0,
                                   UnsatisfiableMessage(hs.core_type, "cube-only (AIC)", "VECTOR"), hs.span);
        } else if (auto msg = OccupancyGateMessage(width, LaunchWidthKind::ClusterCount, total_cube,
                                                   sync_start, "(core_type=\"aic_only\")",
                                                   "fill all " + std::to_string(total_cube) +
                                                       " AIC cores (one block per physical core)")) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0, *msg, hs.span);
        }
      }
      return;
    }

    // Mixed kernel: launched as a Group of {AIC, AIV} sub-kernels. One block maps
    // to one core-group (1 AIC each), so full occupancy fills all core-groups =
    // #CUBE blocks, which fills every AIC and AIV core. This holds for any barrier
    // core_type (mix/aiv_only/aic_only). The hard syncall lives in the sub-kernels
    // (mix is duplicated into both AIC and AIV lanes) — report once per launch.
    if (kernel->func_type_ == FunctionType::Group) {
      auto msg = OccupancyGateMessage(width, LaunchWidthKind::ClusterCount, total_cube, sync_start,
                                      " in mixed kernel '" + kernel->name_ + "'",
                                      "fill all " + std::to_string(total_cube) +
                                          " core-groups (one block per core-group, i.e. all " +
                                          std::to_string(total_cube) + " AIC cores)");
      if (!msg) return;  // both guarantees satisfied
      for (const auto& sub : DirectCallees(kernel, program)) {
        if (!sub || (sub->func_type_ != FunctionType::AIC && sub->func_type_ != FunctionType::AIV)) {
          continue;
        }
        auto hard_syncalls = CollectHardSyncalls(sub);
        if (!hard_syncalls.empty()) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0, *msg,
                                   hard_syncalls.front().span);
          return;
        }
      }
    }
  }
};

PropertyVerifierPtr CreateHardSyncallOccupancyPropertyVerifier() {
  return std::make_shared<HardSyncallOccupancyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
