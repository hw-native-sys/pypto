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

#include "pypto/codegen/code_context.h"

#include <cctype>

#include "pypto/core/logging.h"

namespace pypto {

namespace codegen {

std::string CodeContext::GetVarName(const ir::VarPtr& var) {
  CHECK(var != nullptr) << "Cannot get name for null variable";

  auto it = var_names_.find(var);
  CHECK(it != var_names_.end()) << "Variable " << var->name_ << " not found in context";
  if (it != var_names_.end()) {
    return it->second;
  }

  return "";
}

void CodeContext::RegisterVar(const ir::VarPtr& var, const std::string& cpp_name) {
  CHECK(var != nullptr) << "Cannot register null variable";
  CHECK(!cpp_name.empty()) << "Cannot register variable with empty name";
  CHECK(var_names_.find(var) == var_names_.end()) << "Variable " << var->name_ << " already registered with name " << cpp_name;
  var_names_[var] = cpp_name;
}

void CodeContext::Clear() {
  var_names_.clear();
}

std::string CodeContext::SanitizeName(const ir::VarPtr& var) const {
  CHECK(var != nullptr) << "Cannot sanitize null variable";
  auto ir_name = var->name_;
  if (ir_name.empty()) {
    return "var";
  }

  std::string result;
  result.reserve(ir_name.size());

  // First character must be letter or underscore
  if (std::isalpha(static_cast<unsigned char>(ir_name[0])) || ir_name[0] == '_') {
    result += ir_name[0];
  } else {
    result += '_';
  }

  // Subsequent characters can be alphanumeric or underscore
  for (size_t i = 1; i < ir_name.size(); ++i) {
    char c = ir_name[i];
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      result += c;
    } else {
      result += '_';
    }
  }

  return result;
}

}  // namespace codegen

}  // namespace pypto
