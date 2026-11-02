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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_IR_SELECTION_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_IR_SELECTION_H_

#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/verifier/diagnostic_check_registry.h"

namespace pypto::ir {

// Written once before storage legalization; later passes must not reselect on
// their transformed input. Absence denotes an explicit low-level pass pipeline.
inline bool IsBufferIRSelected(const FunctionPtr& function) {
  return function->GetAttr<bool>(kAttrBufferIrSelected, true);
}

// Keep all storage helpers on the same per-function path, including helpers
// several calls below the function pass entry. The enclosing context and its
// instruments are restored even if legalization or planning throws.
class BufferIRFunctionScope {
 public:
  explicit BufferIRFunctionScope(const FunctionPtr& function)
      : outer_(PassContext::Current()),
        context_(outer_ ? outer_->GetInstruments() : std::vector<PassInstrumentPtr>{},
                 outer_ ? outer_->GetVerificationLevel() : VerificationLevel::Basic,
                 outer_ ? outer_->GetDiagnosticPhase() : DiagnosticPhase::PrePipeline,
                 outer_ ? outer_->GetDisabledDiagnostics()
                        : DiagnosticCheckSet{DiagnosticCheck::UnusedControlFlowResult},
                 outer_ ? outer_->GetMemoryPlanner() : MemoryPlanner::PyPTO,
                 outer_ && outer_->GetEnablePyptoL0cDoubleBuffer(),
                 outer_ ? outer_->GetRuntime() : kDefaultRuntimeKind,
                 (outer_ ? outer_->GetEnableBufferIR() : function->HasAttr(kAttrBufferIrSelected)) &&
                     IsInCoreType(function->func_type_) && IsBufferIRSelected(function),
                 outer_ && outer_->GetEnableSoftwarePipeline()) {
    context_.EnterContext();
  }
  ~BufferIRFunctionScope() { context_.ExitContext(); }
  BufferIRFunctionScope(const BufferIRFunctionScope&) = delete;
  BufferIRFunctionScope& operator=(const BufferIRFunctionScope&) = delete;

 private:
  const PassContext* outer_;
  PassContext context_;
};

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_IR_SELECTION_H_
