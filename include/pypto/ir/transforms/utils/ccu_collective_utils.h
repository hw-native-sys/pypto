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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_CCU_COLLECTIVE_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_CCU_COLLECTIVE_UTILS_H_

#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::ir {

inline constexpr const char* kCcuAllGatherTemplate = ":pypto.runtime.builtins.ccu.allgather";

inline bool IsCcuAllGatherBuiltin(const FunctionPtr& func) {
  return func && func->GetAttr<std::string>(kAttrBuiltinTemplateDir) == kCcuAllGatherTemplate;
}

// HOST numbering and device packaging must canonicalize specialization keys
// identically. Attribute text order is not part of a native definition's identity.
inline std::map<std::string, std::string> GetBuiltinTemplateVariables(const FunctionPtr& func) {
  std::map<std::string, std::string> variables;
  std::istringstream substitutions(func->GetAttr<std::string>(kAttrBuiltinTemplateVars));
  std::string item;
  while (std::getline(substitutions, item, ',')) {
    const auto equal = item.find('=');
    INTERNAL_CHECK_SPAN(equal != std::string::npos && equal > 0 && equal + 1 < item.size(), func->span_)
        << "Invalid builtin specialization: " << item;
    const bool inserted = variables.emplace(item.substr(0, equal), item.substr(equal + 1)).second;
    INTERNAL_CHECK_SPAN(inserted, func->span_) << "Duplicate builtin specialization key: " << item;
  }
  return variables;
}

// Resolve the actual CCU callee from the wrapper body, never a parallel attr.
inline CallPtr CcuWrapperCall(const FunctionPtr& func, const ProgramPtr& program) {
  class Finder : public IRVisitor {
   public:
    explicit Finder(ProgramPtr program) : program_(std::move(program)) {}
    CallPtr call;
    void VisitExpr_(const CallPtr& op) override {
      auto callee = As<GlobalVar>(op->op_) ? program_->GetFunction(op->op_->name_) : nullptr;
      if (callee && callee->func_type_ == FunctionType::CCU) {
        CHECK_SPAN(!call, op->span_) << "CCU wrapper must contain exactly one CCU request";
        call = op;
      }
      IRVisitor::VisitExpr_(op);
    }

   private:
    ProgramPtr program_;
  } finder(program);
  if (func && func->func_type_ == FunctionType::AIV) finder.VisitStmt(func->body_);
  return finder.call;
}

ProgramPtr MaterializeCcuBindings(const ProgramPtr& program);
void VerifyCcuBindings(const ProgramPtr& program);

// CCU signal/order are reference handles with conservative task effects, not
// returned SSA data values. The payload target still obeys ordinary rebind rules.
inline bool IsCcuControlParameter(const FunctionPtr& func, size_t index) {
  return IsCcuAllGatherBuiltin(func) && func->func_type_ == FunctionType::AIV && (index == 2 || index == 3);
}
}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_CCU_COLLECTIVE_UTILS_H_
