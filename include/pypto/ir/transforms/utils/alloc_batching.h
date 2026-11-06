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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ALLOC_BATCHING_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ALLOC_BATCHING_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace alloc_batching {

/// How many `TensorCreateInfo` operands one `alloc_tensors` call carries.
///
/// Orchestration codegen packs eligible `tensor.create`s up to this many per
/// call, and the host_build_graph runtime records one node per call. Both the
/// emitter and `LegalizeGraphBoundary`'s node-limit check reason about how many
/// nodes a region's allocations become, so the rule lives here rather than in
/// either — an estimate that drifts from the emitter is a limit check that
/// rejects legal Graphs or waves through uncacheable ones.
inline constexpr size_t kAllocTensorsArgs = 16;

/// True for a `tensor.create` that `InjectGmPipeBuffer` synthesised.
[[nodiscard]] inline bool IsInjectedGMPipeCreateVar(const VarPtr& var) {
  return var != nullptr && var->name_hint_.rfind("gm_pipe_buffer_", 0) == 0;
}

/// True when @p expr reads any of @p vars.
[[nodiscard]] inline bool ExprRefsAnyOf(const ExprPtr& expr, const std::unordered_set<const Var*>& vars) {
  if (!expr) return false;
  if (auto var = As<Var>(expr)) return vars.count(var.get()) > 0;
  if (auto bin = As<BinaryExpr>(expr)) {
    return ExprRefsAnyOf(bin->left_, vars) || ExprRefsAnyOf(bin->right_, vars);
  }
  if (auto un = As<UnaryExpr>(expr)) return ExprRefsAnyOf(un->operand_, vars);
  if (auto cast_expr = As<Cast>(expr)) return ExprRefsAnyOf(cast_expr->operand_, vars);
  return false;
}

/// The `core_num` a launch effectively carries.
///
/// `pl.spmd_submit` puts it on the dispatch call; a scope-based `pl.spmd` or
/// Group wrapper puts it on the callee. Reading only the callee under-reads a
/// direct `pl.spmd_submit(..., core_num=N)`.
[[nodiscard]] inline ExprPtr EffectiveCoreNum(const CallPtr& call, const FunctionPtr& callee) {
  ExprPtr core_num = call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
  if (!core_num && callee) core_num = callee->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
  return core_num;
}

/// The concrete dispatch whose block-local pipe layout backs one allocation.
struct GMPipeCreateUse {
  FunctionPtr callee;
  ExprPtr core_num;
};

/// Index Graph boundary parameters by their concrete launch consumer.
///
/// A Graph is not itself an SPMD launch. LegalizeGraphBoundary can move a pipe
/// allocation to its caller, so sizing must follow the supplied Graph parameter
/// back to the actual task. Each Graph body is visited once, across all buffers.
/// Names and frontend pipe ids are deliberately not used as storage identity.
class GMPipeConsumerIndex {
 public:
  [[nodiscard]] GMPipeCreateUse Resolve(const CallPtr& graph_call, const FunctionPtr& graph,
                                        const VarPtr& buffer, const ProgramPtr& program) const {
    auto it = graphs_.find(graph.get());
    if (it == graphs_.end()) {
      GraphCollector collector(graph, program, layouts_);
      collector.VisitStmt(graph->body_);
      it = graphs_.emplace(graph.get(), std::move(collector.consumers)).first;
    }
    std::optional<GMPipeCreateUse> result;
    // CallerSuppliedArg excludes omitted Out parameters and the CommCtx suffix.
    const size_t ctx = transform_utils::TrailingCommCtxParamCount(graph);
    INTERNAL_CHECK_SPAN(graph_call->args_.size() >= ctx && graph_call->args_.size() <= graph->params_.size(),
                        graph_call->span_)
        << "Invalid Graph argument coverage while resolving a GM pipe buffer";
    const size_t supplied = graph_call->args_.size() - ctx;
    for (size_t i = 0; i < supplied; ++i) {
      auto actual = AsVarLike(graph_call->args_[i]);
      if (!actual || actual.get() != buffer.get()) continue;
      auto use = it->second.find(graph->params_[i].get());
      INTERNAL_CHECK_SPAN(use != it->second.end(), graph_call->span_)
          << "Graph GM pipe parameter has no concrete launch consumer";
      const auto& concrete_use = use->second.use;
      INTERNAL_CHECK_SPAN(concrete_use.has_value() && !use->second.ambiguous, graph_call->span_)
          << "Graph GM pipe parameter '" << graph->params_[i]->name_hint_
          << "' must have one concrete launch consumer (or identical repeated launches)";
      const auto& candidate = *concrete_use;
      INTERNAL_CHECK_SPAN(!result || SameUse(*result, candidate), graph_call->span_)
          << "One GM pipe allocation cannot bind incompatible Graph launch consumers";
      result = candidate;
    }
    INTERNAL_CHECK_SPAN(result.has_value(), graph_call->span_)
        << "Graph call does not supply the injected GM pipe buffer as a tensor parameter";
    return *result;
  }

 private:
  struct Consumer {
    std::optional<GMPipeCreateUse> use;
    bool ambiguous = false;
  };

  static bool SameUse(const GMPipeCreateUse& a, const GMPipeCreateUse& b) {
    if (a.callee.get() != b.callee.get()) return false;
    const auto a_core = a.core_num ? transform_utils::EvalConstInt(a.core_num) : std::optional<int64_t>(1);
    const auto b_core = b.core_num ? transform_utils::EvalConstInt(b.core_num) : std::optional<int64_t>(1);
    return a_core && b_core && *a_core == *b_core;
  }

  struct CalleeLayout {
    size_t ctx = 0;
    std::optional<size_t> workspace;
  };

  class GraphCollector : public IRVisitor {
   public:
    GraphCollector(const FunctionPtr& graph, ProgramPtr program,
                   std::unordered_map<const Function*, CalleeLayout>& layouts)
        : program_(std::move(program)), layouts_(layouts) {
      for (const auto& param : graph->params_) {
        if (AsTensorTypeLike(param->GetType())) consumers.emplace(param.get(), Consumer{});
      }
    }

    std::unordered_map<const Var*, Consumer> consumers;

   protected:
    void VisitExpr_(const CallPtr& op) override {
      IRVisitor::VisitExpr_(op);
      Record(op, false);
    }

    void VisitExpr_(const SubmitPtr& op) override {
      IRVisitor::VisitExpr_(op);
      Record(SubmitToCallView(op), true);
    }

   private:
    void Record(const CallPtr& call, bool is_submit) {
      auto gv = As<GlobalVar>(call->op_);
      if (!gv) return;
      auto callee = program_->GetFunction(gv->name_);
      if (!callee) return;
      // The injection pass marks the workspace ABI parameter with this
      // reserved name. It identifies a role, never the caller's buffer identity.
      auto [layout_it, inserted] = layouts_.try_emplace(callee.get());
      if (inserted) {
        layout_it->second.ctx = transform_utils::TrailingCommCtxParamCount(callee);
        for (size_t i = 0; i < callee->params_.size(); ++i) {
          if (callee->params_[i]->name_hint_ == "__gm_pipe_buffer") {
            layout_it->second.workspace = i;
            break;
          }
        }
      }
      const auto& layout = layout_it->second;
      if (!layout.workspace) return;
      const bool coverage_ok = is_submit ? call->args_.size() <= callee->params_.size()
                                         : call->args_.size() == callee->params_.size();
      INTERNAL_CHECK_SPAN(coverage_ok && call->args_.size() >= layout.ctx, call->span_)
          << "Invalid call argument coverage while resolving Graph GM pipe consumers";
      // This is CallerSuppliedArg's prefix mapping, with the suffix count
      // cached per callee. The omitted Out gap and CommCtx suffix are excluded.
      const size_t supplied = call->args_.size() - layout.ctx;
      if (*layout.workspace >= supplied) return;
      auto arg = AsVarLike(call->args_[*layout.workspace]);
      if (!arg) return;
      auto found = consumers.find(arg.get());
      if (found == consumers.end()) return;
      const bool concrete =
          callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd ||
          callee->func_type_ == FunctionType::AIC || callee->func_type_ == FunctionType::AIV;
      auto core = EffectiveCoreNum(call, callee);
      auto literal = core ? transform_utils::EvalConstInt(core) : std::optional<int64_t>(1);
      auto& consumer = found->second;
      if (!concrete || !literal || *literal <= 0) {
        consumer.ambiguous = true;
        return;
      }
      GMPipeCreateUse use{callee, core};
      if (consumer.use && !SameUse(*consumer.use, use)) consumer.ambiguous = true;
      if (!consumer.use) consumer.use = std::move(use);
    }

    ProgramPtr program_;
    std::unordered_map<const Function*, CalleeLayout>& layouts_;
  };

  mutable std::unordered_map<const Function*, std::unordered_map<const Var*, Consumer>> graphs_;
  mutable std::unordered_map<const Function*, CalleeLayout> layouts_;
};

/// Concrete consumer and effective launch width for one injected allocation,
/// or nullopt when there is no such launch.
///
/// Walks forward to the first call taking the buffer, exactly as the emitter
/// does, and stops at a rebind of the same Var identity.
[[nodiscard]] inline std::optional<GMPipeCreateUse> ResolveGMPipeCreateUse(
    const std::vector<StmtPtr>& stmts, size_t create_stmt_idx, const VarPtr& create_var,
    const ProgramPtr& program, const GMPipeConsumerIndex& consumers) {
  if (!create_var || !program) return std::nullopt;
  for (size_t i = create_stmt_idx + 1; i < stmts.size(); ++i) {
    auto assign = As<AssignStmt>(stmts[i]);
    if (assign && assign->var_ && assign->var_.get() == create_var.get()) break;
    CallPtr call;
    if (assign) {
      call = transform_utils::AsCallOrSubmitView(assign->value_);
    } else if (auto eval = As<EvalStmt>(stmts[i])) {
      call = transform_utils::AsCallOrSubmitView(eval->expr_);
    }
    if (!call) continue;
    bool uses_buffer = false;
    for (const auto& arg : call->args_) {
      auto var = AsVarLike(arg);
      if (var && var.get() == create_var.get()) {
        uses_buffer = true;
        break;
      }
    }
    if (!uses_buffer) continue;
    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return std::nullopt;
    auto callee = program->GetFunction(gv->name_);
    if (!callee) return std::nullopt;
    if (callee->func_type_ == FunctionType::Graph) {
      return consumers.Resolve(call, callee, create_var, program);
    }
    // Preserve the ordinary eager launch source and its dynamic expressions.
    return GMPipeCreateUse{callee, EffectiveCoreNum(call, callee)};
  }
  return std::nullopt;
}

/// True when the GM pipe buffer at @p create_stmt_idx keeps its place in the
/// shared batch.
///
/// The emitter pulls one out when its `core_num` reads a value defined earlier
/// in the same statement list, because the create then has to stay in body
/// order. That is the single allocation-batching decision a caller cannot make
/// from the create alone, which is why it is resolved here for both of them.
[[nodiscard]] inline bool GMPipeCreateJoinsBatch(const std::vector<StmtPtr>& stmts, size_t create_stmt_idx,
                                                 const VarPtr& create_var, const ProgramPtr& program,
                                                 const std::unordered_set<const Var*>& locally_defined,
                                                 const GMPipeConsumerIndex& consumers) {
  auto use = ResolveGMPipeCreateUse(stmts, create_stmt_idx, create_var, program, consumers);
  auto core_num = use ? use->core_num : nullptr;
  return !ExprRefsAnyOf(core_num, locally_defined);
}

/// Nodes that @p creates batchable allocations become.
[[nodiscard]] inline size_t BatchedAllocationNodes(size_t creates) {
  return (creates + kAllocTensorsArgs - 1) / kAllocTensorsArgs;
}

}  // namespace alloc_batching
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ALLOC_BATCHING_H_
