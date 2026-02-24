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

#include "pypto/backend/common/backend_registry.h"

#include <memory>
#include <string>
#include <utility>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/soc.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"

namespace pypto {
namespace backend {

BackendRegistry& BackendRegistry::Instance() {
  static BackendRegistry instance;
  return instance;
}

void BackendRegistry::Register(const std::string& type_name, CreateFunc func) {
  CHECK(registry_.find(type_name) == registry_.end()) << "Backend type already registered: " << type_name;
  registry_[type_name] = std::move(func);
}

std::unique_ptr<Backend> BackendRegistry::Create(const std::string& type_name,
                                                 const std::shared_ptr<const SoC>& soc) {
  // For singleton backends, we cannot create new instances
  throw ValueError(
      "Cannot create backend instances via registry - backends are singletons. "
      "Use Backend910B_CCE::Instance() or Backend910B_PTO::Instance() instead.");
}

bool BackendRegistry::IsRegistered(const std::string& type_name) const {
  return registry_.find(type_name) != registry_.end();
}

std::unique_ptr<Backend> CreateBackendFromRegistry(const std::string& type_name,
                                                   const std::shared_ptr<const SoC>& soc) {
  // For singleton backends, we cannot create new instances
  throw ValueError(
      "Cannot create backend instances via registry - backends are singletons. "
      "Use Backend910B_CCE::Instance() or Backend910B_PTO::Instance() instead.");
}

// Auto-register Backend910B_CCE and Backend910B_PTO
namespace {
bool RegisterBackend910B_CCE() {
  // Backend910B_CCE is a singleton, no need to register factory function
  // Registration is kept for backward compatibility but Create() will fail
  BackendRegistry::Instance().Register("910B_CCE", [](const std::shared_ptr<const SoC>& /*unused*/) {
    throw ValueError("Cannot create Backend910B_CCE via registry - use Backend910B_CCE::Instance()");
    return std::unique_ptr<Backend>(nullptr);  // Never reached
  });
  return true;
}

bool RegisterBackend910B_PTO() {
  // Backend910B_PTO is a singleton, no need to register factory function
  // Registration is kept for backward compatibility but Create() will fail
  BackendRegistry::Instance().Register("910B_PTO", [](const std::shared_ptr<const SoC>& /*unused*/) {
    throw ValueError("Cannot create Backend910B_PTO via registry - use Backend910B_PTO::Instance()");
    return std::unique_ptr<Backend>(nullptr);  // Never reached
  });
  return true;
}

static bool backend_910b_cce_registered = RegisterBackend910B_CCE();
static bool backend_910b_pto_registered = RegisterBackend910B_PTO();
}  // namespace

}  // namespace backend
}  // namespace pypto
