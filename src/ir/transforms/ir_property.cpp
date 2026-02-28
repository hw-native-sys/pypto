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

#include "pypto/ir/transforms/ir_property.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace pypto {
namespace ir {

std::string IRPropertyToString(IRProperty prop) {
  switch (prop) {
    case IRProperty::SSAForm:
      return "SSAForm";
    case IRProperty::TypeChecked:
      return "TypeChecked";
    case IRProperty::NoNestedCalls:
      return "NoNestedCalls";
    case IRProperty::NormalizedStmtStructure:
      return "NormalizedStmtStructure";
    case IRProperty::FlattenedSingleStmt:
      return "FlattenedSingleStmt";
    case IRProperty::SplitIncoreOrch:
      return "SplitIncoreOrch";
    case IRProperty::HasMemRefs:
      return "HasMemRefs";
    case IRProperty::IncoreBlockOps:
      return "IncoreBlockOps";
    case IRProperty::AllocatedMemoryAddr:
      return "AllocatedMemoryAddr";
    default:
      return "Unknown";
  }
}

std::vector<IRProperty> IRPropertySet::ToVector() const {
  std::vector<IRProperty> result;
  for (uint32_t i = 0; i < static_cast<uint32_t>(IRProperty::kCount); ++i) {
    auto prop = static_cast<IRProperty>(i);
    if (Contains(prop)) {
      result.push_back(prop);
    }
  }
  return result;
}

std::string IRPropertySet::ToString() const {
  auto props = ToVector();
  if (props.empty()) {
    return "{}";
  }

  std::ostringstream oss;
  oss << "{";
  for (size_t i = 0; i < props.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << IRPropertyToString(props[i]);
  }
  oss << "}";
  return oss.str();
}

const IRPropertySet& GetVerifiedProperties() {
  static const IRPropertySet props{IRProperty::SSAForm, IRProperty::TypeChecked,
                                   IRProperty::AllocatedMemoryAddr};
  return props;
}

VerificationLevel GetDefaultVerificationLevel() {
  // C++17: static local initialization is thread-safe and runs exactly once
  static const VerificationLevel level = [] {
    const char* env = std::getenv("PYPTO_VERIFY_LEVEL");
    if (env == nullptr) {
      return VerificationLevel::Basic;
    }
    std::string val(env);
    if (val == "none") {
      return VerificationLevel::None;
    }
    return VerificationLevel::Basic;
  }();
  return level;
}

}  // namespace ir
}  // namespace pypto
