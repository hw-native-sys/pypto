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

/**
 * @file warning_out_param_write_dropped.cpp
 * @brief Warn when rebinding an Out/InOut parameter drops the caller's write (issue #2352).
 *
 * `out = pl.add(a, b)` rebinds the Python name only: the parameter Var is
 * re-pointed at a freshly computed tensor and the caller's buffer is never
 * written. It compiles and runs with no diagnostic anywhere, and the caller
 * gets back uninitialised memory (`Out`, freshly allocated) or its own unchanged
 * input (`InOut`). The write must go through the parameter —
 * `out[:] = pl.add(a, b)`.
 *
 * Distinguishing a dropped write from a legitimate rebind needs data flow, not
 * syntax. `out = pl.assemble(out, x, [0, 0])` mentions the parameter directly,
 * but a value can also reach it through a loop carry:
 *
 *     for col, (d,) in pl.range(0, n, c, init_values=(data,)):
 *         d = pl.store(local, [0, col], d)
 *         staged = pl.yield_(d)
 *     data = pld.tensor.allreduce(staged, signal, ...)   # writes through `data`
 *
 * `staged` never names `data`, yet it *is* `data` threaded through the carry.
 * This check therefore taints each Out/InOut parameter forward through the
 * definition graph and only warns when the assigned value shares nothing with
 * that taint.
 *
 * Deliberately conservative: a value that merely *reads* the parameter
 * (`out = pl.add(out, b)`) is tainted too, so it is not reported even though it
 * also drops the write. Separating "reads the parameter" from "writes through
 * the parameter" needs per-op write semantics the registry does not record —
 * `pld.tensor.allreduce` is `.no_memory_spec()`, for one — and a false warning
 * on correct collective code is worse than a missed one.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/diagnostic_check_registry.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Warning error code (1000+ range for warnings; see warning_unused_var.cpp).
constexpr int kOutParamWriteDroppedCode = 1003;

using VarSet = std::unordered_set<const Var*>;

/// Collect every Var read inside an expression, including nested Calls and a
/// Submit's `deps_` (see the pass-submit-awareness rule).
class ExprVarCollector : public IRVisitor {
 public:
  VarSet vars;

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) vars.insert(op.get());
  }
};

VarSet CollectVars(const ExprPtr& expr) {
  ExprVarCollector collector;
  if (expr) collector.VisitExpr(expr);
  return collector.vars;
}

/// One assignment that rebinds an Out/InOut parameter.
struct ParamRebind {
  const Var* param;
  VarSet value_vars;
  Span span;
};

/**
 * @brief Builds the per-function definition graph and records parameter rebinds.
 *
 * `defined_by_` maps each defined Var to the Vars its definition reads. A Var
 * defined more than once (the pre-SSA IR reuses one Var across reassignments)
 * accumulates the union of every definition, which over-approximates the taint
 * and therefore only ever suppresses a warning.
 *
 * Loop and branch results are handled with the same union: every iter_arg and
 * every return_var of a region depends on all of that region's init values and
 * yielded values. Modelling the exact per-index carry would narrow the taint,
 * but narrowing it risks false positives on correct code.
 */
class DefinitionGraphBuilder : public IRVisitor {
 public:
  explicit DefinitionGraphBuilder(const VarSet& tracked_params) : tracked_params_(tracked_params) {}

  std::unordered_map<const Var*, VarSet> defined_by;
  std::vector<ParamRebind> rebinds;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op || !op->var_) return;
    auto value_vars = CollectVars(op->value_);
    if (tracked_params_.count(op->var_.get()) > 0) {
      rebinds.push_back(ParamRebind{op->var_.get(), value_vars, op->span_});
    }
    AddDefinition(op->var_.get(), value_vars);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    IRVisitor::VisitStmt_(op);
    AddRegionCarry(op->iter_args_, op->return_vars_, op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    IRVisitor::VisitStmt_(op);
    AddRegionCarry(op->iter_args_, op->return_vars_, op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    IRVisitor::VisitStmt_(op);
    VarSet sources = CollectYieldedVars(op->then_body_);
    if (op->else_body_.has_value()) {
      VarSet else_vars = CollectYieldedVars(*op->else_body_);
      sources.insert(else_vars.begin(), else_vars.end());
    }
    for (const auto& rv : op->return_vars_) {
      if (rv) AddDefinition(rv.get(), sources);
    }
  }

 private:
  /// Collects the Vars every YieldStmt in `body` yields, without descending
  /// into nested *loop* regions — those bind their own results, which the
  /// visitor already recorded on the way in.
  ///
  /// A nested `IfStmt` is deliberately not stopped: were a loop's carry yield
  /// ever to sit inside a branch, skipping it would drop the carry from the
  /// taint and turn a correct write-through into a false warning. Attributing a
  /// branch's yields to the enclosing region instead only widens the taint,
  /// which can only suppress a warning.
  class YieldVarCollector : public IRVisitor {
   public:
    VarSet vars;

   protected:
    void VisitExpr(const ExprPtr& /*expr*/) override {}

    void VisitStmt_(const YieldStmtPtr& op) override {
      if (!op) return;
      for (const auto& value : op->value_) {
        VarSet value_vars = CollectVars(value);
        vars.insert(value_vars.begin(), value_vars.end());
      }
    }

    void VisitStmt_(const ForStmtPtr& /*op*/) override {}
    void VisitStmt_(const WhileStmtPtr& /*op*/) override {}
  };

  static VarSet CollectYieldedVars(const StmtPtr& body) {
    YieldVarCollector collector;
    if (body) collector.VisitStmt(body);
    return collector.vars;
  }

  void AddRegionCarry(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars,
                      const StmtPtr& body) {
    VarSet sources = CollectYieldedVars(body);
    for (const auto& iter_arg : iter_args) {
      if (!iter_arg) continue;
      VarSet init_vars = CollectVars(iter_arg->initValue_);
      sources.insert(init_vars.begin(), init_vars.end());
    }
    for (const auto& iter_arg : iter_args) {
      if (iter_arg) AddDefinition(iter_arg.get(), sources);
    }
    for (const auto& rv : return_vars) {
      if (rv) AddDefinition(rv.get(), sources);
    }
  }

  void AddDefinition(const Var* var, const VarSet& sources) {
    auto& entry = defined_by[var];
    entry.insert(sources.begin(), sources.end());
  }

  const VarSet& tracked_params_;
};

/// Number of roots one `ComputeTaintMasks` call can carry, one bit each.
constexpr size_t kTaintBatchSize = 64;

/// For every Var, the bitmask of `roots` it is transitively derived from.
///
/// The reverse index is built once and a single worklist pass propagates all
/// roots together, so the cost is O(V + E) for the whole batch rather than per
/// root. Masks only ever grow, so the fixpoint terminates after at most one
/// push per (Var, bit).
std::unordered_map<const Var*, uint64_t> ComputeTaintMasks(
    const std::vector<const Var*>& roots, const std::unordered_map<const Var*, VarSet>& defined_by) {
  INTERNAL_CHECK(roots.size() <= kTaintBatchSize)
      << "Internal error: taint batch carries " << roots.size() << " roots, capacity is " << kTaintBatchSize;

  // Reverse index: source Var -> Vars whose definition reads it.
  std::unordered_map<const Var*, std::vector<const Var*>> readers;
  for (const auto& [defined, sources] : defined_by) {
    for (const auto* source : sources) {
      readers[source].push_back(defined);
    }
  }

  std::unordered_map<const Var*, uint64_t> masks;
  std::vector<const Var*> worklist;
  for (size_t i = 0; i < roots.size(); ++i) {
    masks[roots[i]] |= uint64_t{1} << i;
    worklist.push_back(roots[i]);
  }

  while (!worklist.empty()) {
    const auto* current = worklist.back();
    worklist.pop_back();
    const uint64_t current_mask = masks[current];
    auto it = readers.find(current);
    if (it == readers.end()) continue;
    for (const auto* reader : it->second) {
      uint64_t& reader_mask = masks[reader];
      const uint64_t merged = reader_mask | current_mask;
      if (merged != reader_mask) {
        reader_mask = merged;
        worklist.push_back(reader);
      }
    }
  }
  return masks;
}

/// The remedy only exists for shaped parameters — `out[:] = <expr>` has no
/// scalar spelling — so the check is scoped to Tensor / Tile parameters.
bool IsShapedParam(const VarPtr& param) { return param && As<ShapedType>(param->GetType()) != nullptr; }

class OutParamWriteDroppedVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "OutParamWriteDropped"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      if (func && func->body_) VerifyFunction(func, diagnostics);
    }
  }

 private:
  static void VerifyFunction(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
    // Tracked parameters in signature order (so batch membership is stable),
    // plus the direction lookup the message needs — building the map here keeps
    // the emit path off a per-diagnostic scan of the parameter list. Within a
    // batch, diagnostics follow statement order.
    std::vector<const Var*> tracked;
    VarSet tracked_set;
    std::unordered_map<const Var*, ParamDirection> direction_by_param;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      const auto& param = func->params_[i];
      const auto direction = func->param_directions_[i];
      const bool writable = direction == ParamDirection::Out || direction == ParamDirection::InOut;
      if (writable && IsShapedParam(param)) {
        tracked.push_back(param.get());
        tracked_set.insert(param.get());
        direction_by_param[param.get()] = direction;
      }
    }
    if (tracked.empty()) return;

    DefinitionGraphBuilder builder(tracked_set);
    builder.VisitStmt(func->body_);
    if (builder.rebinds.empty()) return;

    // Every parameter is tainted in one shared pass over one shared reverse
    // index. A signature wider than the batch takes one extra pass per 64
    // parameters, which keeps the check linear in the size of the function
    // body rather than O(params x body).
    for (size_t base = 0; base < tracked.size(); base += kTaintBatchSize) {
      const size_t end = std::min(base + kTaintBatchSize, tracked.size());
      const std::vector<const Var*> roots(tracked.begin() + static_cast<std::ptrdiff_t>(base),
                                          tracked.begin() + static_cast<std::ptrdiff_t>(end));
      std::unordered_map<const Var*, size_t> bit_of_param;
      for (size_t i = 0; i < roots.size(); ++i) {
        bit_of_param[roots[i]] = i;
      }
      const auto masks = ComputeTaintMasks(roots, builder.defined_by);

      for (const auto& rebind : builder.rebinds) {
        auto bit_it = bit_of_param.find(rebind.param);
        if (bit_it == bit_of_param.end()) continue;  // handled by another batch
        const uint64_t param_bit = uint64_t{1} << bit_it->second;

        const bool flows_through_param =
            std::any_of(rebind.value_vars.begin(), rebind.value_vars.end(), [&](const Var* v) {
              auto mask_it = masks.find(v);
              return mask_it != masks.end() && (mask_it->second & param_bit) != 0;
            });
        if (flows_through_param) continue;

        std::ostringstream msg;
        msg << "Assigning to " << DirectionName(direction_by_param, rebind.param) << " parameter '"
            << rebind.param->name_hint_ << "' in function '" << func->name_
            << "' rebinds the name only — the caller's buffer is never written. Use '"
            << rebind.param->name_hint_ << "[:] = <expr>' to write the whole tensor, or '"
            << rebind.param->name_hint_ << "[<slices>] = <expr>' for a sub-window.";
        diagnostics.emplace_back(DiagnosticSeverity::Warning, "OutParamWriteDropped",
                                 kOutParamWriteDroppedCode, msg.str(), rebind.span);
      }
    }
  }

  static const char* DirectionName(const std::unordered_map<const Var*, ParamDirection>& directions,
                                   const Var* param) {
    auto it = directions.find(param);
    INTERNAL_CHECK(it != directions.end()) << "Internal error: rebind names an untracked parameter";
    return it->second == ParamDirection::Out ? "Out" : "InOut";
  }
};

}  // namespace

PropertyVerifierPtr CreateOutParamWriteDroppedWarningVerifier() {
  return std::make_shared<OutParamWriteDroppedVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
