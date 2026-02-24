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

#include "pypto/ir/transforms/op_conversion_registry.h"

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

OpConversionRegistry& OpConversionRegistry::GetInstance() {
  static OpConversionRegistry instance;
  return instance;
}

OpConversionRegistry::OpConversionRegistry() {
  // Register default simple conversions (tensor op -> block op)

  // Elementwise binary ops
  RegisterSimple("tensor.add", "block.add");
  RegisterSimple("tensor.sub", "block.sub");
  RegisterSimple("tensor.mul", "block.mul");
  RegisterSimple("tensor.div", "block.div");
  RegisterSimple("tensor.maximum", "block.maximum");

  // Scalar ops
  RegisterSimple("tensor.add_scalar", "block.adds");
  RegisterSimple("tensor.sub_scalar", "block.subs");
  RegisterSimple("tensor.mul_scalar", "block.muls");
  RegisterSimple("tensor.div_scalar", "block.divs");

  // Unary ops
  RegisterSimple("tensor.exp", "block.exp");
  RegisterSimple("tensor.cast", "block.cast");

  // Transform ops
  RegisterSimple("tensor.reshape", "block.reshape");
  RegisterSimple("tensor.transpose", "block.transpose");
}

void OpConversionRegistry::RegisterSimple(const std::string& from_op, const std::string& to_op) {
  // Capture to_op by value for the lambda
  conversions_[from_op] = [to_op](const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const Span& span) -> ConversionResult {
    auto& reg = OpRegistry::GetInstance();
    CallPtr call;
    if (kwargs.empty()) {
      call = reg.Create(to_op, args, span);
    } else {
      call = reg.Create(to_op, args, kwargs, span);
    }
    return ConversionResult{call};
  };
}

void OpConversionRegistry::RegisterCustom(const std::string& from_op, ConversionFunc func) {
  conversions_[from_op] = std::move(func);
}

const ConversionFunc* OpConversionRegistry::Lookup(const std::string& op_name) const {
  auto it = conversions_.find(op_name);
  if (it == conversions_.end()) {
    return nullptr;
  }
  return &it->second;
}

bool OpConversionRegistry::HasConversion(const std::string& op_name) const {
  return conversions_.count(op_name) > 0;
}

}  // namespace ir
}  // namespace pypto
