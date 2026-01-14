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

#include "pypto/ir/op_registry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"

namespace pypto {
namespace ir {

OpRegistry& OpRegistry::GetInstance() {
  static OpRegistry instance;
  return instance;
}

OpRegistryEntry& OpRegistry::Register(const std::string& op_name) {
  // Check if operator is already registered
  CHECK(registry_.find(op_name) == registry_.end()) << "Operator '" + op_name + "' is already registered";

  // Create and insert the entry into the registry
  auto result = registry_.emplace(op_name, OpRegistryEntry());
  auto& entry = result.first->second;
  entry.set_name(op_name);

  // Create the operator instance with the operator name
  entry.op_ = std::make_shared<Op>(op_name);

  return entry;
}

// ============================================================================
// OpRegistry Implementation
// ============================================================================

CallPtr OpRegistry::Create(const std::string& op_name, const std::vector<ExprPtr>& args, Span span) const {
  // Look up operator in registry
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";

  const auto& entry = it->second;

  // Get operator instance and type deduction function (validation is done inside getters)
  OpPtr op = entry.GetOp();
  const auto& deduce_type_fn = entry.GetDeduceType();

  // Deduce result type
  TypePtr result_type = deduce_type_fn(args);
  INTERNAL_CHECK(result_type) << "Type deduction failed for '" + op_name + "'";

  // Create Call expression with deduced type
  auto call = std::make_shared<Call>(op, args, result_type, std::move(span));
  return call;
}

OpPtr OpRegistry::GetOp(const std::string& op_name) const {
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";
  return it->second.GetOp();
}

}  // namespace ir
}  // namespace pypto
