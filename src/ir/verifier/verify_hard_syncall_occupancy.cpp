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
// its `core_type` to arrive at the barrier. If the enclosing `pl.spmd(N)` launch
// does not fill all those cores, the unlaunched cores never reach the barrier,
// the FFTS wait never completes, and the AICore times out on device (507018),
// leaving it needing a reset. The soft (GM-polling) form exists for partial
// occupancy. This verifier turns that runtime footgun into a compile-time error.
//
// It runs after ExpandMixedKernel, where each launched kernel's FunctionType is
// resolved (AIV / AIC / Group), which is what lets us map spmd blocks to physical
// cores per launch shape:
//   - Spmd -> standalone AIV kernel  : 1 block = 1 AIV core  -> required = #VECTOR
//   - Spmd -> standalone AIC kernel  : 1 block = 1 AIC core  -> required = #CUBE
//   - Spmd -> Group (mixed kernel)   : 1 block = 1 core-group (1 AIC each) ->
//                                      required = #CUBE (fills all core-groups,
//                                      hence all AIC and all AIV)
//
// A bare hard-syncall kernel with no `pl.spmd` launch is not checked: occupancy
// is a launch-time property.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
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

// Resolve the functions directly called by `fn` (callees only — op calls such as
// tile.load resolve to nullptr via GetFunction and are dropped). Handles both
// Call (scope-based launch) and Submit (spmd_submit), per pass-submit-awareness.
std::vector<FunctionPtr> DirectCallees(const FunctionPtr& fn, const ProgramPtr& program) {
  CallSubmitCollector collector;
  if (fn->body_) collector.VisitStmt(fn->body_);
  std::vector<FunctionPtr> out;
  auto add = [&](const OpPtr& op) {
    if (!op) return;
    if (auto callee = program->GetFunction(op->name_)) out.push_back(callee);
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

std::string BuildMessage(const std::string& detail, int64_t launched, const std::string& requirement) {
  return "hard pl.system.syncall" + detail + " requires the enclosing pl.spmd launch to " + requirement +
         ", but pl.spmd launches " + std::to_string(launched) +
         " blocks. Use mode=\"soft\" (GM-polling) for partial occupancy.";
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

    for (const auto& [gv, spmd_fn] : program->functions_) {
      if (!spmd_fn || spmd_fn->func_type_ != FunctionType::Spmd) continue;
      auto core_num_expr = spmd_fn->GetAttr<ExprPtr>("core_num", nullptr);
      if (!core_num_expr) continue;
      auto core_num_const = As<ConstInt>(core_num_expr);
      if (!core_num_const) continue;  // dynamic launch count — cannot check statically
      const int64_t launched = core_num_const->value_;

      for (const auto& callee : DirectCallees(spmd_fn, program)) {
        CheckLaunchedKernel(callee, launched, total_vector, total_cube, program, diagnostics);
      }
    }
  }

 private:
  static void CheckLaunchedKernel(const FunctionPtr& kernel, int64_t launched, int total_vector,
                                  int total_cube, const ProgramPtr& program,
                                  std::vector<Diagnostic>& diagnostics) {
    if (!kernel) return;

    // Standalone AIV kernel: 1 block = 1 AIV core (aiv_only barrier).
    // A dual-AIV-dispatched AIV kernel is a mixed-kernel lane reached via Group,
    // not a standalone launch — handled by the Group branch instead.
    if (kernel->func_type_ == FunctionType::AIV && !kernel->HasAttr("dual_aiv_dispatch")) {
      for (const auto& hs : CollectHardSyncalls(kernel)) {
        if (hs.core_type == "aiv_only" && launched != total_vector) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0,
                                   BuildMessage("(core_type=\"aiv_only\")", launched,
                                                "fill all " + std::to_string(total_vector) +
                                                    " AIV cores (one block per physical core)"),
                                   hs.span);
        }
      }
      return;
    }

    // Standalone AIC kernel: 1 block = 1 AIC core (aic_only barrier).
    if (kernel->func_type_ == FunctionType::AIC) {
      for (const auto& hs : CollectHardSyncalls(kernel)) {
        if (hs.core_type == "aic_only" && launched != total_cube) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "HardSyncallOccupancy", 0,
                                   BuildMessage("(core_type=\"aic_only\")", launched,
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
                                   BuildMessage(" in mixed kernel '" + kernel->name_ + "'", launched,
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
