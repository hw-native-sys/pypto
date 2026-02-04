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

#include "pypto/backend/backend_registry.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pypto/backend/backend_910b.h"
#include "pypto/core/logging.h"

namespace pypto {
namespace backend {

BackendRegistry& BackendRegistry::Instance() {
  static BackendRegistry instance;
  return instance;
}

void BackendRegistry::Register(const std::string& type_name, CreateFunc func) {
  CHECK(registry_.find(type_name) == registry_.end()) << "Backend type already registered: " << type_name;
  registry_[type_name] = func;
}

std::unique_ptr<Backend> BackendRegistry::Create(
    const std::string& type_name, std::shared_ptr<const SoC> soc,
    const std::map<ir::MemorySpace, std::vector<ir::MemorySpace>>& mem_graph) {
  auto it = registry_.find(type_name);
  CHECK(it != registry_.end()) << "Unknown backend type: " << type_name;
  return it->second(soc, mem_graph);
}

bool BackendRegistry::IsRegistered(const std::string& type_name) const {
  return registry_.find(type_name) != registry_.end();
}

std::unique_ptr<Backend> CreateBackendFromRegistry(
    const std::string& type_name, std::shared_ptr<const SoC> soc,
    const std::map<ir::MemorySpace, std::vector<ir::MemorySpace>>& mem_graph) {
  return BackendRegistry::Instance().Create(type_name, soc, mem_graph);
}

// Auto-register Backend910B
namespace {
bool RegisterBackend910B() {
  BackendRegistry::Instance().Register(
      "910B", [](std::shared_ptr<const SoC> /*unused*/,
                 std::map<ir::MemorySpace, std::vector<ir::MemorySpace>> /*unused*/) {
        // Backend910B creates its own SoC and memory hierarchy, ignore parameters
        return std::make_unique<Backend910B>();
      });
  return true;
}

static bool backend_910b_registered = RegisterBackend910B();
}  // namespace

}  // namespace backend
}  // namespace pypto
