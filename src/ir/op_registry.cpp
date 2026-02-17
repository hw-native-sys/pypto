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

#include <any>
#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

void ValidateKwargs(const std::vector<std::pair<std::string, std::any>>& kwargs,
                    const std::unordered_map<std::string, std::type_index>& allowed_kwargs,
                    const std::string& op_name) {
  for (const auto& [key, value] : kwargs) {
    auto it = allowed_kwargs.find(key);
    if (it == allowed_kwargs.end()) {
      throw ValueError("Unknown kwarg '" + key + "' for operator '" + op_name + "'");
    }

    // For DataType, accept both DataType and int (since Python may pass as int for backward compatibility)
    if (it->second == std::type_index(typeid(DataType))) {
      std::type_index value_type(value.type());
      if (value_type != std::type_index(typeid(DataType)) && value_type != std::type_index(typeid(int))) {
        throw TypeError("Kwarg '" + key + "' for operator '" + op_name +
                        "' expects DataType or int, but got incompatible type");
      }
    } else if (std::type_index(value.type()) != it->second) {
      throw TypeError("Kwarg '" + key + "' for operator '" + op_name + "' has incompatible type");
    }
  }
}

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
  // Call new version with empty kwargs for backward compatibility
  return Create(op_name, args, {}, std::move(span));
}

CallPtr OpRegistry::Create(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, Span span) const {
  // Look up operator in registry
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";

  const auto& entry = it->second;

  // Get operator instance (shared definition)
  OpPtr op = entry.GetOp();

  // Validate kwargs against allowed attributes (stored in Op)
  if (!kwargs.empty()) {
    const auto& allowed_kwargs = op->GetAttrs();
    if (!allowed_kwargs.empty()) {
      ValidateKwargs(kwargs, allowed_kwargs, op_name);
    }
  }

  const auto& deduce_type_fn = entry.GetDeduceType();

  // Deduce result type (pass args and kwargs separately)
  TypePtr result_type = deduce_type_fn(args, kwargs);
  INTERNAL_CHECK(result_type) << "Type deduction failed for '" + op_name + "'";

  // Create Call with deduced type
  return std::make_shared<Call>(op, args, kwargs, result_type, std::move(span));
}

const OpRegistryEntry& OpRegistry::GetEntry(const std::string& op_name) const {
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";
  return it->second;
}

OpPtr OpRegistry::GetOp(const std::string& op_name) const {
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";
  return it->second.GetOp();
}

}  // namespace ir
}  // namespace pypto
