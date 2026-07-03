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
// its `core_type` to arrive at the barrier. If the enclosing SPMD launch does
// not fill all those cores, the unlaunched cores never reach the barrier, the
// FFTS wait never completes, and the AICore times out on device (507018),
// leaving it needing a reset. The soft (GM-polling) form exists for partial
// occupancy. This verifier turns that runtime footgun into a compile-time error.
//
// It runs after ExpandMixedKernel, where each launched kernel's FunctionType is
// resolved (AIV / AIC / Group), which is what lets us map SPMD blocks to physical
// cores per launch shape. It covers every SPMD launch site that carries a
// compile-time-constant block count:
//   - `FunctionType::Spmd` function  (scope-based `with/for pl.spmd`)          [core_num attr]
//   - `FunctionType::Group` function with a core_num attr (`pl.cluster()`-nested
//     `pl.spmd`, unwrapped by OutlineClusterScopes)                            [core_num attr]
//   - a `Submit` node with a core_num (`pl.spmd_submit(..., core_num=N)`)      [Submit::core_num_]
//
// Given a launch site's block count N and its direct callee kernel K:
//   - K standalone AIV  + aiv_only  -> required = #VECTOR (1 block = 1 AIV core)
//   - K standalone AIC  + aic_only  -> required = #CUBE   (1 block = 1 AIC core)
//   - K standalone AIV/AIC + a *different* core_type (incl. the default `mix`) ->
//     unsatisfiable: the other core type has zero participants in a single-core
//     launch, so the barrier can never complete regardless of N.
//   - K Group (mixed AIC+AIV kernel) -> required = #CUBE core-groups (1 AIC each),
//     for any barrier core_type; the hard syncall lives in the AIC/AIV sub-kernels
//     (a `mix` barrier is duplicated into both lanes -> reported once).
// N != required is an error (partial *and* over-occupancy).
//
// A bare hard-syncall kernel with no SPMD launch is not checked: occupancy is a
// launch-time property.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
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

    for (const auto& [gv, fn] : program->functions_) {
      if (!fn) continue;

      // Launch-site functions: a Spmd wrapper (scope-based pl.spmd), or a Group
      // carrying a core_num attr (a pl.cluster()-nested pl.spmd unwrapped by
      // OutlineClusterScopes). A Group *without* core_num is a mixed-kernel
      // wrapper (a launch callee), handled inside CheckLaunchedKernel.
      const bool is_launch_fn = fn->func_type_ == FunctionType::Spmd ||
                                (fn->func_type_ == FunctionType::Group && fn->HasAttr("core_num"));
      if (is_launch_fn) {
        CheckLaunchSite(fn->GetAttr<ExprPtr>("core_num", nullptr), DirectCallees(fn, program), total_vector,
                        total_cube, program, diagnostics);
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
        CheckLaunchSite(*submit->core_num_, {callee}, total_vector, total_cube, program, diagnostics);
      }
    }
  }

 private:
  static void CheckLaunchSite(const ExprPtr& core_num_expr, const std::vector<FunctionPtr>& callees,
                              int total_vector, int total_cube, const ProgramPtr& program,
                              std::vector<Diagnostic>& diagnostics) {
    if (!core_num_expr) return;
    auto core_num_const = As<ConstInt>(core_num_expr);
    if (!core_num_const) return;  // dynamic launch count — cannot check statically
    const int64_t launched = core_num_const->value_;
    for (const auto& callee : callees) {
      CheckLaunchedKernel(callee, launched, total_vector, total_cube, program, diagnostics);
    }
  }

  static void CheckLaunchedKernel(const FunctionPtr& kernel, int64_t launched, int total_vector,
                                  int total_cube, const ProgramPtr& program,
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
        } else if (launched != total_vector) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0,
                                   OccupancyMessage("(core_type=\"aiv_only\")", launched,
                                                    "fill all " + std::to_string(total_vector) +
                                                        " AIV cores (one block per physical core)"),
                                   hs.span);
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
        } else if (launched != total_cube) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0,
                                   OccupancyMessage("(core_type=\"aic_only\")", launched,
                                                    "fill all " + std::to_string(total_cube) +
                                                        " AIC cores (one block per physical core)"),
                                   hs.span);
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
      if (launched == total_cube) return;
      for (const auto& sub : DirectCallees(kernel, program)) {
        if (!sub || (sub->func_type_ != FunctionType::AIC && sub->func_type_ != FunctionType::AIV)) {
          continue;
        }
        auto hard_syncalls = CollectHardSyncalls(sub);
        if (!hard_syncalls.empty()) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0,
                                   OccupancyMessage(" in mixed kernel '" + kernel->name_ + "'", launched,
                                                    "fill all " + std::to_string(total_cube) +
                                                        " core-groups (one block per core-group, i.e. all " +
                                                        std::to_string(total_cube) + " AIC cores)"),
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
