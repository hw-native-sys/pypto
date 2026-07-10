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

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
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
  // materialization, and a stamped iter_arg carry plan on every ForStmt.
  //
  // ReturnParamsExplicit is listed because codegen *reads the IR directly* for
  // the return->param map: it takes each callee's returned param off the
  // ReturnStmt by pointer identity (NormalizeReturnOrder establishes that form)
  // instead of tracing SSA lineage. Without it, an SSA-aliased return would
  // silently alias a result to the wrong buffer rather than raise.
  //
  // TODO(call-directions-precondition): CallDirectionsResolved belongs here
  // too — codegen equally trusts `callee->param_directions_`, which
  // DeriveCallDirections materializes on Group/Spmd wrappers. It is not listed
  // yet because two existing tests feed orchestration codegen IR that
  // deliberately violates it (an `Input` arg direction on an `Out` param, and a
  // convert_to_ssa-only program), so wiring it needs those tests reworked.
  pass::VerifyProperties(
      IRPropertySet{IRProperty::SplitIncoreOrch, IRProperty::OrchestrationReferencesResolved,
                    IRProperty::RuntimeScopesMaterialized, IRProperty::IterArgCarryClassified,
                    IRProperty::ReturnParamsExplicit},
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
