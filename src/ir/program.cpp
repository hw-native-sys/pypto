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

#include "pypto/ir/program.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"

namespace pypto {
namespace ir {

// Vector-based constructor: creates GlobalVars from function names
Program::Program(const std::vector<FunctionPtr>& functions, std::string name, Span span)
    : IRNode(std::move(span)), name_(std::move(name)) {
  // Create a map and populate it with GlobalVar -> Function mappings
  // The map automatically sorts by GlobalVar name via the GlobalVarPtrLess comparator
  std::set<std::string> function_names;
  for (const auto& func : functions) {
    INTERNAL_CHECK(func) << "Program constructor encountered null function";
    auto name = func->name_;
    INTERNAL_CHECK(!name.empty()) << "Program constructor encountered empty function name";
    CHECK(function_names.find(name) == function_names.end()) << "Duplicate function name \"" << name << "\"";
    function_names.insert(name);
    auto global_var = std::make_shared<const GlobalVar>(name);
    functions_.emplace(global_var, func);
  }
}

FunctionPtr Program::GetFunction(const std::string& name) const {
  auto it = functions_.find(std::make_shared<const GlobalVar>(name));
  if (it != functions_.end()) {
    return it->second;
  }
  return nullptr;
}

GlobalVarPtr Program::GetGlobalVar(const std::string& name) const {
  auto it = functions_.find(std::make_shared<const GlobalVar>(name));
  if (it != functions_.end()) {
    return it->first;
  }
  return nullptr;
}

}  // namespace ir
}  // namespace pypto
