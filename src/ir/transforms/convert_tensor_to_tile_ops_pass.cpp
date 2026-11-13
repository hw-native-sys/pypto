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

#include <cstddef>
#include <memory>
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
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/op_conversion_registry.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/buffer_root_collector.h"
#include "pypto/ir/transforms/utils/call_shape_binding.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/narrow_loop_carry.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/verifier/verifier.h"
#include "src/ir/transforms/tensor_to_tile/internal.h"

namespace pypto {
namespace ir {

using tensor_to_tile::AddedOutputBindings;
using tensor_to_tile::AddedOutputMap;
using tensor_to_tile::CallShapePlans;
using tensor_to_tile::PropagateOutputsThroughWrapper;
using tensor_to_tile::TransformIncoreFunction;
using tensor_to_tile::UpdateCallSites;

namespace {

// Conversion and its postcondition must cover exactly the same kernel bodies.
bool ShouldConvertTensorToTileOps(const FunctionPtr& func) {
  return func && func->body_ && func->ir_stage_ == FunctionIRStage::Functional &&
         IsInCoreType(func->func_type_) && !func->HasAttr(kAttrExternalSource) &&
         !func->requires_runtime_binding_;
}

}  // namespace

namespace pass {

Pass ConvertTensorToTileOps() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 1: explicit AIC/AIV kernels use the same output materialization
    // as generic InCore kernels, while keeping their selected core type.
    AddedOutputMap all_added_outputs;
    std::unordered_map<std::string, FunctionPtr> all_transformed_funcs;
    std::vector<FunctionPtr> functions_phase1;
    CallShapePlans shape_plans;
    for (const auto& [gvar, func] : program->functions_) {
      shape_plans.emplace(func->name_, CallShapeBindingPlan(func));
      // Buffer kernels already carry explicit storage and separate data /
      // metadata effects; functional tensor-to-tile conversion must leave
      // their bodies and signatures intact.
      if (ShouldConvertTensorToTileOps(func)) {
        auto result = TransformIncoreFunction(func);
        all_added_outputs.emplace(func->name_, std::move(result.added_outputs));
        all_transformed_funcs.emplace(func->name_, result.func);
        functions_phase1.push_back(result.func);
      } else {
        all_added_outputs.emplace(func->name_, AddedOutputBindings{});
        all_transformed_funcs.emplace(func->name_, func);
        functions_phase1.push_back(func);
      }
    }

    // Phase 2a: build the wrapper call graph once, then propagate in postorder.
    // An outer Spmd must see its Group's new outputs regardless of name order.
    class WrapperCalleeCollector : public IRVisitor {
     public:
      std::vector<std::string> names;

     protected:
      void VisitExpr_(const CallPtr& op) override {
        Record(op->op_);
        IRVisitor::VisitExpr_(op);
      }
      void VisitExpr_(const SubmitPtr& op) override {
        Record(op->op_);
        IRVisitor::VisitExpr_(op);
      }

     private:
      void Record(const OpPtr& op) {
        if (auto global = As<GlobalVar>(op); global && seen_.insert(global->name_).second) {
          names.push_back(global->name_);
        }
      }
      std::unordered_set<std::string> seen_;
    };
    std::unordered_map<std::string, std::vector<std::string>> wrapper_callees;
    for (const auto& func : functions_phase1) {
      if (func->ir_stage_ == FunctionIRStage::Buffer || !IsWrapperType(func->func_type_)) continue;
      WrapperCalleeCollector collector;
      collector.VisitStmt(func->body_);
      wrapper_callees.emplace(func->name_, std::move(collector.names));
    }
    std::unordered_map<std::string, int> wrapper_state;
    auto propagate_wrapper = [&](auto&& self, const std::string& name) -> void {
      auto edges = wrapper_callees.find(name);
      if (edges == wrapper_callees.end() || wrapper_state[name] == 2) return;
      const auto& func = all_transformed_funcs.at(name);
      CHECK_SPAN(wrapper_state[name] != 1, func->span_)
          << "Recursive Group/Spmd wrappers cannot propagate output parameters";
      wrapper_state[name] = 1;
      for (const auto& callee : edges->second) self(self, callee);
      auto result =
          PropagateOutputsThroughWrapper(func, all_added_outputs, all_transformed_funcs, shape_plans);
      all_added_outputs[name] = std::move(result.added_outputs);
      all_transformed_funcs[name] = result.func;
      wrapper_state[name] = 2;
    };
    for (const auto& func : functions_phase1) propagate_wrapper(propagate_wrapper, func->name_);

    // Phase 2b: allocate each exposed output at the orchestration call site.
    std::vector<FunctionPtr> functions_phase2b;
    functions_phase2b.reserve(functions_phase1.size());
    for (const auto& source : functions_phase1) {
      const auto& func = all_transformed_funcs.at(source->name_);
      if (func->ir_stage_ == FunctionIRStage::Buffer || IsInCoreType(func->func_type_) ||
          IsWrapperType(func->func_type_)) {
        functions_phase2b.push_back(func);
      } else {
        functions_phase2b.push_back(
            UpdateCallSites(func, all_added_outputs, all_transformed_funcs, shape_plans));
      }
    }

    // Phase 3: Propagate Function::param_directions_ along the call chain.
    //
    // When the user writes inline `pl.at(...)` blocks, OutlineHierarchyScopes
    // extracts them into a host_orch → chip_orch → incore chain. The outlined
    // chip_orch has no direction info on its own parameters yet. Phase 1 has
    // already marked the InCore's tile-written params as Out/InOut; if
    // chip_orch(a, b, f) forwards its own `f` to that InCore, chip_orch's
    // own `f` must be upgraded to Out so the signature matches the data flow.
    //
    // This phase mutates Function::param_directions_ (function signature)
    // only. Per-call-site arg directions (Call::attrs_["arg_directions"]) are
    // owned by the later DeriveCallDirections pass and are not touched here.
    // Analyze callees first so provisional In parameters in wrappers do not
    // count as content reads before the wrappers' directions are resolved.
    {
      std::unordered_map<std::string, size_t> function_indices;
      for (size_t i = 0; i < functions_phase2b.size(); ++i) {
        function_indices.emplace(functions_phase2b[i]->name_, i);
      }

      auto effect_program = std::make_shared<Program>(functions_phase2b, program->name_, program->span_);

      struct CallerFacts {
        std::unordered_map<const Var*, size_t> param_idx;
        std::vector<CallPtr> calls;
        std::vector<CallPtr> builtins;
        std::unordered_map<const Var*, const Var*> buffer_roots;
      };
      std::unordered_map<std::string, CallerFacts> facts_by_name;
      for (const auto& func : functions_phase2b) {
        if (func->ir_stage_ == FunctionIRStage::Buffer || func->func_type_ == FunctionType::InCore) continue;

        CallerFacts f;
        for (size_t i = 0; i < func->params_.size(); ++i) {
          f.param_idx[func->params_[i].get()] = i;
        }

        class CallScanner : public IRVisitor {
         public:
          std::vector<CallPtr> calls;
          std::vector<CallPtr> builtins;
          void VisitExpr_(const CallPtr& call) override {
            Record(call);
            IRVisitor::VisitExpr_(call);
          }

          /// A task launch forwards its arguments exactly as a plain call does,
          /// and the base visitor does not route `Submit` through the `Call`
          /// handler (`.claude/rules/pass-submit-awareness.md`). Without this a
          /// parameter handed to an `Out` callee through `pl.submit` never
          /// reached the propagation below, so an orchestration function that
          /// only ever submits kept declaring `In` for a buffer its tasks write.
          /// The view is transient — the loop reads only `op_` and `args_` — and
          /// `args_` is a positional prefix of the callee's params, which the
          /// loop's dual bound already respects.
          void VisitExpr_(const SubmitPtr& submit) override {
            Record(SubmitToCallView(submit));
            IRVisitor::VisitExpr_(submit);
          }

         private:
          void Record(const CallPtr& call) {
            if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
              calls.push_back(call);
            } else {
              builtins.push_back(call);
            }
          }
        };
        CallScanner scanner;
        scanner.VisitStmt(func->body_);
        f.calls = std::move(scanner.calls);
        f.builtins = std::move(scanner.builtins);

        facts_by_name[func->name_] = std::move(f);
      }

      return_lineage::ScopedTraceCache lineage_cache;
      buffer_root::ReturnedRootCandidates returned_candidates;
      std::vector<bool> visiting(functions_phase2b.size(), false);
      std::vector<bool> resolved(functions_phase2b.size(), false);
      const auto infer = [&](auto&& self, size_t index) -> void {
        if (resolved[index]) return;
        auto& func = functions_phase2b[index];
        const auto facts_it = facts_by_name.find(func->name_);
        if (facts_it == facts_by_name.end()) return;
        CHECK_SPAN(!visiting[index], func->span_) << "Recursive call graph cannot infer parameter directions";
        visiting[index] = true;
        for (const auto& call : facts_it->second.calls) {
          const auto callee = function_indices.find(call->op_->name_);
          if (callee != function_indices.end()) self(self, callee->second);
        }
        auto& facts = facts_it->second;
        buffer_root::BufferRootCollector roots(effect_program, buffer_root::AmbiguousRootPolicy::kTraceReturn,
                                               /*follow_reshape=*/true, &returned_candidates);
        roots.Initialize(func->params_);
        roots.VisitStmt(func->body_);
        auto result_candidates = roots.CollectAmbiguousReturnedParams(func);
        facts.buffer_roots = std::move(roots.buffer_roots);
        std::vector<bool> reads(func->params_.size(), false);
        std::vector<bool> writes(func->params_.size(), false);
        auto new_dirs = func->param_directions_;
        for (const auto& call : facts.calls) {
          const auto callee_it = function_indices.find(call->op_->name_);
          if (callee_it == function_indices.end()) continue;
          const auto& directions = functions_phase2b[callee_it->second]->param_directions_;
          for (size_t ai = 0; ai < call->args_.size() && ai < directions.size(); ++ai) {
            auto arg = AsVarLike(call->args_[ai]);
            if (!arg || !AsTensorTypeLike(arg->GetType())) continue;
            const auto direction = directions[ai];
            const auto record_access = [&](const Var* root) {
              const auto param = facts.param_idx.find(root);
              if (param == facts.param_idx.end()) return;
              const size_t i = param->second;
              if (direction != ParamDirection::Out) reads[i] = true;
              if (direction != ParamDirection::In) writes[i] = true;
              if (direction == ParamDirection::InOut) new_dirs[i] = ParamDirection::InOut;
            };
            if (roots.ambiguous_buffer_vars.count(arg.get())) {
              for (const auto* root : roots.RootCandidatesOf(arg.get())) record_access(root);
            } else {
              const auto root = facts.buffer_roots.find(arg.get());
              record_access(root == facts.buffer_roots.end() ? arg.get() : root->second);
            }
          }
        }
        // Direct orchestration reads consume data. View construction and other
        // descriptor operations do not.
        for (const auto& call : facts.builtins) {
          if (!IsOp(call, "tensor.read") || call->args_.empty()) continue;
          const auto arg = AsVarLike(call->args_[0]);
          if (!arg) continue;
          const auto mark_read = [&](const Var* root) {
            const auto it = facts.param_idx.find(root);
            if (it != facts.param_idx.end()) reads[it->second] = true;
          };
          if (roots.ambiguous_buffer_vars.count(arg.get())) {
            for (const auto* root : roots.RootCandidatesOf(arg.get())) mark_read(root);
          } else {
            const auto root = facts.buffer_roots.find(arg.get());
            mark_read(root == facts.buffer_roots.end() ? arg.get() : root->second);
          }
        }
        for (size_t i = 0; i < new_dirs.size(); ++i) {
          if (new_dirs[i] == ParamDirection::In && writes[i]) {
            new_dirs[i] = reads[i] ? ParamDirection::InOut : ParamDirection::Out;
          }
        }
        resolved[index] = true;
        if (new_dirs != func->param_directions_) {
          auto updated = MutableCopy(func);
          updated->param_directions_ = std::move(new_dirs);
          effect_program->functions_.at(effect_program->GetGlobalVar(func->name_)) = updated;
          func = std::move(updated);
        }
        if (!result_candidates.empty()) {
          returned_candidates.emplace(func.get(), std::move(result_candidates));
        }
      };
      for (size_t i = 0; i < functions_phase2b.size(); ++i) infer(infer, i);
    }

    // 2D tensor.matmul already propagates valid M/N, but carries are still typed
    // from their seeds. Conversion also introduces physical boxing and Acc
    // layout/compact metadata. Reconcile converted yields with their carries
    // before TypeCheck and AccCompactValid run (issue #2470).
    for (auto& func : functions_phase2b) {
      if (func->ir_stage_ == FunctionIRStage::Buffer) continue;
      func = narrow_loop_carry::NarrowAccCarries(func);
    }

    return std::make_shared<Program>(functions_phase2b, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ConvertTensorToTileOps", kConvertTensorToTileOpsProperties);
}

}  // namespace pass

// ============================================================================
// IncoreTileOps property verifier
// ============================================================================

namespace {

/**
 * @brief Checks converted InCore/AIC/AIV functions for surviving tensor ops.
 */
class IncoreTileOpsVerifier : public IRVisitor {
 public:
  explicit IncoreTileOpsVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckTensorOp(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckTensorOp(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckTensorOp(const std::shared_ptr<const Call>& call, const Span& span) {
    // Op calls use plain Op (not GlobalVar); GlobalVar is for function calls
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (global_var) return;

    // Use op category from OpRegistry instead of brittle string prefix check
    auto& op_registry = OpRegistry::GetInstance();
    if (!op_registry.IsRegistered(call->op_->name_)) return;

    const auto& entry = op_registry.GetEntry(call->op_->name_);
    if (entry.GetOpCategory() == "TensorOp" &&
        OpConversionRegistry::GetInstance().HasConversion(call->op_->name_)) {
      // tensor.read/tensor.write on a gm_tensor (TensorType input) intentionally stays unconverted.
      // ``AsTensorTypeLike`` also whitelists ``DistributedTensorType``, which the
      // conversion registry above keeps as ``tensor.read`` / ``tensor.write`` so the
      // PTO codegen can lower it as a local-rank ``pto.load`` / ``pto.store``.
      if ((IsOp(call, "tensor.read") || IsOp(call, "tensor.write")) && !call->args_.empty() &&
          AsTensorTypeLike(call->args_[0]->GetType())) {
        return;
      }

      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "IncoreTileOps", 0,
          "Tensor op '" + call->op_->name_ + "' found in InCore function (should have been converted)", span);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class IncoreTileOpsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "IncoreTileOps"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!ShouldConvertTensorToTileOps(func)) continue;
      IncoreTileOpsVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateIncoreTileOpsPropertyVerifier() {
  return std::make_shared<IncoreTileOpsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
