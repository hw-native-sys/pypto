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
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
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
      void VisitStmt_(const SimtLaunchStmtPtr& op) override {
        Record(op->callee_);
        IRVisitor::VisitStmt_(op);
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

    // Phase 3: merge storage reads and writes through the call graph.
    // An Out writer does not make a caller write-only when a separate In
    // reader consumes the same (possibly reshaped) cache. Analyze callees first:
    // their initial In signatures may still become pure Out, so a fixed point
    // that permanently latches those initial In reads invents dependencies.
    //
    // The call graph and each body are walked once. The private program below
    // is updated only after a callee's directions are final; no source Program
    // or body is mutated, and lookups remain O(log F).
    {
      auto effect_program = std::make_shared<Program>(functions_phase2b, program->name_, program->span_);
      std::unordered_map<std::string, size_t> positions;
      std::unordered_map<std::string, std::vector<std::string>> callees;
      for (size_t i = 0; i < functions_phase2b.size(); ++i) {
        const auto& func = functions_phase2b[i];
        positions.emplace(func->name_, i);
        if (func->ir_stage_ == FunctionIRStage::Buffer || IsInCoreType(func->func_type_)) continue;
        WrapperCalleeCollector collector;
        collector.VisitStmt(func->body_);
        callees.emplace(func->name_, std::move(collector.names));
      }

      buffer_root::ScopedStorageRootCache storage_cache;
      std::unordered_map<std::string, int> states;
      auto propagate = [&](auto&& self, const std::string& name) -> void {
        auto edges = callees.find(name);
        if (edges == callees.end() || states[name] == 2) return;
        auto& func = functions_phase2b.at(positions.at(name));
        CHECK_SPAN(states[name] != 1, func->span_)
            << "Recursive call graph cannot infer storage parameter directions";
        states[name] = 1;
        for (const auto& callee : edges->second) self(self, callee);

        auto directions = outline_utils::InferStorageParamDirections(func->params_, func->body_,
                                                                     effect_program, func->param_directions_);
        states[name] = 2;
        if (directions != func->param_directions_) {
          auto updated = MutableCopy(func);
          updated->param_directions_ = std::move(directions);
          // name may borrow the old function. Finish name lookups before
          // replacing the vector's final reference to that function.
          effect_program->functions_.at(effect_program->GetGlobalVar(name)) = updated;
          func = std::move(updated);
        }
      };
      for (const auto& func : functions_phase2b) propagate(propagate, func->name_);
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
