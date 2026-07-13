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

#include "pypto/codegen/codegen_preconditions.h"

#include <string>

#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

namespace {

class DistributedTensorUseCollector : public IRVisitor {
 public:
  bool uses_distributed_tensor = false;
  bool has_comm_domain_scope = false;
  bool has_windowing_ops = false;

 protected:
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    has_comm_domain_scope = true;
    uses_distributed_tensor = true;
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarPtr& var) override {
    if (As<DistributedTensorType>(var->GetType())) {
      uses_distributed_tensor = true;
    }
    IRVisitor::VisitExpr_(var);
  }

  void VisitExpr_(const CallPtr& call) override {
    const std::string& op_name = call->op_->name_;
    if (op_name.rfind("pld.tensor.", 0) == 0 || op_name.rfind("tensor.window", 0) == 0 ||
        op_name.rfind("tensor.alloc_window_buffer", 0) == 0 ||
        op_name.rfind("tensor.window_buffer", 0) == 0 || As<DistributedTensorType>(call->GetType())) {
      uses_distributed_tensor = true;
      has_windowing_ops = true;
    }
    IRVisitor::VisitExpr_(call);
  }
};

}  // namespace

void VerifyOrchestrationCodegenPreconditions(const ProgramPtr& program, const FunctionPtr& func) {
  INTERNAL_CHECK(program != nullptr)
      << "Internal error: GenerateOrchestration preconditions — program must not be null";
  INTERNAL_CHECK(func != nullptr)
      << "Internal error: GenerateOrchestration preconditions — function must not be null";

  // Codegen assumes hierarchy references resolved, explicit RuntimeScopeStmt
  // materialization, a stamped iter_arg carry plan on every ForStmt, and the
  // manual-scope allocation-hoist set marked on every hoistable tensor.create.
  pass::VerifyProperties(
      IRPropertySet{IRProperty::SplitIncoreOrch, IRProperty::OrchestrationReferencesResolved,
                    IRProperty::RuntimeScopesMaterialized, IRProperty::IterArgCarryClassified,
                    IRProperty::HoistableAllocsMarked},
      program, "GenerateOrchestration preconditions");
}

void VerifyDistributedCodegenPreconditions(const ProgramPtr& program) {
  INTERNAL_CHECK(program != nullptr)
      << "Internal error: DistributedCodegen preconditions — program must not be null";

  DistributedTensorUseCollector collector;
  collector.VisitProgram(program);
  if (!collector.uses_distributed_tensor) {
    return;
  }

  INTERNAL_CHECK(!collector.has_windowing_ops || collector.has_comm_domain_scope)
      << "Internal error: DistributedCodegen preconditions — MaterializeCommDomainScopes must run before "
         "DistributedCodegen when window-buffer/distributed-tensor ops are present. "
         "The pass pipeline is incomplete.";

  // Comm-domain materialization is required when DistributedTensor values are
  // present in host orchestration paths.
  pass::VerifyProperties(IRPropertySet{IRProperty::CommDomainScopesMaterialized}, program,
                         "DistributedCodegen preconditions");
}

}  // namespace codegen
}  // namespace pypto
