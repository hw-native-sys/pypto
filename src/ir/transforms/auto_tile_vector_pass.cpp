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

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "src/ir/transforms/auto_tile/vector_emit.h"
#include "src/ir/transforms/auto_tile/vector_plan.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

class CalledFunctionCollector : public IRVisitor {
 public:
  explicit CalledFunctionCollector(const Program* program) : program_(program) {}

  [[nodiscard]] const std::unordered_set<std::string>& called() const { return called_; }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (call != nullptr && call->op_ != nullptr && program_->GetFunction(call->op_->name_) != nullptr)
      called_.insert(call->op_->name_);
    IRVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const SubmitPtr& submit) override {
    if (submit != nullptr && submit->op_ != nullptr && program_->GetFunction(submit->op_->name_) != nullptr)
      called_.insert(submit->op_->name_);
    IRVisitor::VisitExpr_(submit);
  }

 private:
  const Program* program_;
  std::unordered_set<std::string> called_;
};

struct VectorTargetContext {
  auto_tile::VectorHardware hardware;
  const backend::BackendHandler* handler = nullptr;
};

VectorTargetContext ReadVectorTarget(const Span& span) {
  CHECK_SPAN(backend::BackendConfig::IsConfigured(), span)
      << "AutoTile requires an explicitly configured backend";
  const backend::Backend* backend = backend::GetBackend();
  const PassContext* context = PassContext::Current();
  const backend::BackendHandler* handler =
      context != nullptr ? context->GetBackendHandler() : backend->GetHandler();
  CHECK_SPAN(handler != nullptr, span) << "AutoTile could not obtain the active backend handler";
  const std::optional<backend::VectorAutoTileTarget> target = handler->GetVectorAutoTileTarget();
  CHECK_SPAN(target.has_value(), span) << "AutoTile vector scheduling currently supports Ascend910B only";
  VectorTargetContext result;
  result.handler = handler;
  result.hardware.vector_cores = backend->GetCoreCount(CoreType::VECTOR);
  result.hardware.ub_bytes = static_cast<int64_t>(backend->GetMemSize(MemorySpace::Vec));
  result.hardware.dma_alignment_bytes = target->dma_alignment_bytes;
  result.hardware.vector_register_bytes = target->vector_register_bytes;
  CHECK_SPAN(result.hardware.vector_cores > 0 && result.hardware.ub_bytes > 0 &&
                 result.hardware.dma_alignment_bytes > 0 && result.hardware.vector_register_bytes > 0,
             span)
      << "AutoTile could not derive the " << target->model_name << " vector topology";
  return result;
}

ProgramPtr TransformAutoTileVector(const ProgramPtr& program) {
  if (program == nullptr) return program;
  CalledFunctionCollector calls(program.get());
  for (const auto& [unused, function] : program->functions_) {
    (void)unused;
    if (function != nullptr && function->body_ != nullptr) calls.VisitStmt(function->body_);
  }

  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions;
  bool changed = false;
  for (const auto& [global, function] : program->functions_) {
    if (function == nullptr || !function->GetAttr<bool>("auto_tile", false)) {
      functions.emplace(global, function);
      continue;
    }
    CHECK_SPAN(function->func_type_ == FunctionType::Opaque, function->span_)
        << "AutoTile must run on a tensor-level Opaque function before scope outlining";
    const VectorTargetContext target = ReadVectorTarget(function->span_);
    auto_tile::VectorAdmissionResult admission =
        auto_tile::AdmitVectorGraph(function, program, target.handler->GetTcvtAdjacency());
    if (!admission.supported && admission.failure != nullptr) std::rethrow_exception(admission.failure);
    CHECK_SPAN(admission.supported, function->span_) << "AutoTile cannot admit the entire marked function";
    const auto_tile::VectorGraph& graph = admission.graph;
    const auto_tile::VectorSchedulePlan plan = auto_tile::VectorPlanner910B(target.hardware).Plan(graph);
    CHECK_SPAN(plan.feasible, function->span_)
        << "AutoTile cannot realize the entire marked function as one capacity-safe Ascend910B vector kernel";
    FunctionPtr emitted = auto_tile::EmitVectorSchedule(graph, plan, calls.called());
    LOG_INFO << "AutoTile[" << function->name_
             << "]: vector schedule=" << auto_tile::ScheduleKindName(plan.kind)
             << " grid=" << plan.m_partition.parts << "x" << plan.n_partition.parts
             << " work_units=" << plan.work_units << " tile=" << plan.tile_h << "x" << plan.tile_w
             << " strip=" << plan.strip_h << "x" << plan.strip_w << " chunks=" << plan.full_chunks << "x"
             << plan.chunk << "+" << plan.tail
             << " stages=" << plan.phases[auto_tile::PhaseIndex(auto_tile::VectorPhase::Body)].pipeline_stages
             << "/" << plan.phases[auto_tile::PhaseIndex(auto_tile::VectorPhase::Stats)].pipeline_stages
             << "/" << plan.phases[auto_tile::PhaseIndex(auto_tile::VectorPhase::Apply)].pipeline_stages
             << " peak_ub=" << plan.chunk_peak_ub_bytes << " full_peak_ub=" << plan.full_peak_ub_bytes
             << " compute_cycles=" << plan.modeled_compute_cycles
             << " transfer_cycles=" << plan.modeled_transfer_cycles
             << " modeled_cycles=" << plan.modeled_cycles;
    functions.emplace(global, emitted);
    changed = true;
  }
  if (!changed) return program;
  auto result = MutableCopy(program);
  result->functions_ = std::move(functions);
  return result;
}

}  // namespace

Pass AutoTile() { return CreateProgramPass(TransformAutoTileVector, "AutoTile", kAutoTileProperties); }

}  // namespace pass
}  // namespace ir
}  // namespace pypto
